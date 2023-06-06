import numpy as np
import tensorflow as tf

from sub_models import CNNModel, MultiHeadAttentionModel, QLogitModel

from battlefield_strategy import BattleFieldStrategy
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_po_id_mask as make_mask


class MarlTransformerDecentralized(tf.keras.models.Model):

    def __init__(self, config, cnn_model, multihead_attention_model, qlogit_model):
        super(MarlTransformerDecentralized, self).__init__()

        self.config = config

        self.cnn = cnn_model(self.config)
        self.mha1 = multihead_attention_model(self.config)
        self.mha2 = multihead_attention_model(self.config)
        self.mha3 = multihead_attention_model(self.config)
        self.mha4 = multihead_attention_model(self.config)
        self.qlogit = qlogit_model(self.config)

    def mha_block(self, mha, features, masks, training=True):
        """
        :param mha: mha1, mha2, mha3, mha4
        :param features: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
        :param masks: [(b,1,n),...,(b,1,n)], len=n=15
        :return: att_features: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
                 att_scores: [(b,num_heads,n),...,(b,num_heads,n)], len=n=15
        """
        att_features = []
        att_scores = []

        for i in range(self.config.max_num_red_agents):
            query_feature = features[i]  # (b,hidden_dim)=(1,64)
            agent_inputs = [query_feature, features]
            mask = masks[i]  # (b,1,n)=(1,1,n)

            att_feature, att_score = mha(
                agent_inputs, mask, training)  # (b,hidden_dim)=(1,64), (b,num_heads,n)=(1,2,15)

            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            att_feature = att_feature * broadcast_float_mask  # (b,hidden_dim)=(1,64)

            broadcast_float_mask = tf.cast(
                tf.expand_dims(mask[:, :, i], axis=1),
                'float32')  # (b,1,1)=(1,1,1), add head_dim
            att_score = att_score * broadcast_float_mask  # (b,num_heads,n)=(1,2,15)

            att_features.append(att_feature)
            att_scores.append(att_score)

        return att_features, att_scores

    @tf.function
    def call(self, inputs, masks, training=True):
        """
        :param inputs: [[s1,pos1],...], len=n=15,
            si: (b,2*fov+1,2*fov+1,ch*n_frames)=(1,5,5,16), posi:(b,2*n_frames)=(1,8)
        :param masks: [(b,1,n),...]=[(1,1,15),...], len=n
        :return: qlogits [(b,action_dim),...,(b,action_dim)], len=n
                 attscores = [attscores_1, attscores_4],
                    attscores_i: [(b,num_heads,n),...,(b,num_heads,n)], len=n
        """

        """ CNN block """
        features = []
        for i in range(self.config.max_num_red_agents):
            feature = self.cnn(inputs[i])  # (b,hidden_dim)=(1,64)

            mask = masks[i]  # (b,1,n)=(1,1,n)

            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            feature = feature * broadcast_float_mask  # (b,hidden_dim)=(1,4)

            features.append(feature)

        """ Transformer block """
        # att_features_1, _2, _3, _4: [(b,hidden_dim),...,(b,hidden_dim)], len=n=15
        # att_scores_1, _2, _3, _4: [(b,num_heads,n),...,(b,num_heads,n)], len=n
        # attscores_i: [(b, num_heads, n), ..., (b, num_heads, n)], len=n

        att_features_1, att_scores_1 = self.mha_block(self.mha1, features, masks, training)
        att_features_2, att_scores_2 = self.mha_block(self.mha2, att_features_1, masks, training)
        att_features_3, att_scores_3 = self.mha_block(self.mha3, att_features_2, masks, training)
        att_features_4, att_scores_4 = self.mha_block(self.mha4, att_features_3, masks, training)

        att_scores = [att_scores_1, att_scores_4]

        """ Q logits block """
        q_logits = []
        for i in range(self.config.max_num_red_agents):
            q_logit = self.qlogit(att_features_4[i])  # (None,action_dim)=(1,5)

            mask = masks[i]  # (b,1,n)
            broadcast_float_mask = tf.cast(mask[:, :, i], 'float32')  # (b,1)=(1,1)
            q_logit = q_logit * broadcast_float_mask  # (None,action_dim)=(1,5)

            q_logits.append(q_logit)

        return q_logits, att_scores


def main():
    env = BattleFieldStrategy()
    env.reset()
    config = env.config

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    fov = config.fov
    com = config.com

    max_num_agents = config.max_num_red_agents  # 15

    state_shape = (2 * fov + 1, 2 * fov + 1, ch * n_frames)

    # Define alive_agents_ids
    alive_agents_ids = get_alive_agents_ids(env=env)

    # States
    padded_states = []
    for i in range(config.max_num_red_agents):
        if i in alive_agents_ids:
            state = \
                np.random.rand(state_shape[0], state_shape[1], state_shape[2]).astype(np.float32)
            pos = np.random.rand(2*n_frames).astype(np.float32)
        else:
            state = np.zeros((state_shape[0], state_shape[1], state_shape[2])).astype(np.float32)
            pos = np.zeros((2*n_frames,)).astype(np.float32)

        state = np.expand_dims(state, axis=0)  # (1,5,5,16)
        pos = np.expand_dims(pos, axis=0)  # (1,8)
        padded_states.append([state, pos])

    # Make mask
    masks = make_mask(alive_agents_ids,
                      max_num_agents,
                      env.reds,
                      com)  # [(1,1,n),...]=[(1,1,15),...],len=15

    """ MARL Transformer """
    marl_transformer = MarlTransformerDecentralized(config,
                                                    CNNModel,
                                                    MultiHeadAttentionModel,
                                                    QLogitModel)

    """ Execute MARL Transformer """
    training = True
    q_logits, att_scores = marl_transformer(padded_states, masks, training)

    marl_transformer.summary()

    print(len(q_logits))
    print(q_logits[0].shape)
    print(len(att_scores))
    print(len(att_scores[0]))
    print(att_scores[0][0].shape)


if __name__ == '__main__':
    main()
