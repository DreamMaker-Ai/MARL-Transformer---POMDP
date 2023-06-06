import os.path

import tensorflow as tf
import numpy as np

from battlefield_strategy import BattleFieldStrategy
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_po_id_mask as make_mask


class CNNModel(tf.keras.models.Model):
    """
    :param inputs: [s,pos]=[(b,5,5,16),(b,2)]
    :return: (None,hidden_dim)=(None,64)
    """

    def __init__(self, config, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv1 = \
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv2 = \
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.flatten1 = tf.keras.layers.Flatten()

        self.dense_pos_enc = \
            tf.keras.layers.Dense(
                units=32,
                activation='relu'
            )

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.dense1 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation=None
            )

    @tf.function
    def call(self, inputs):
        # inputs: [s,pos]
        #   s: (b,2*fov+1,2*fov+1,ch*n_frames)=(1,5,5,16)
        #   pos: (b,2*n_frames)=(1,8)

        h = self.conv0(inputs[0])  # (1,5,5,64)
        h = self.conv1(h)  # (1,3,3,128)
        h = self.conv2(h)  # (1,1,1,128)

        h = self.flatten1(h)  # (1,128)

        pos_enc = self.dense_pos_enc(inputs[1])  # (1,32)

        z = self.concatenate([h, pos_enc])  # (1,160)
        features = self.dense1(z)  # (1,64)

        return features


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,1,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 64

    :return: features: (None,hidden_dim)=(None,64)
             score: (None,num_heads,n)=(None,2,15)
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.query_feature = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.expand_dims(x, axis=1), dtype=tf.float32)
            )

        self.features = \
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.stack(x, axis=1), dtype=tf.float32)
            )

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        self.dense = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        self.reshape = tf.keras.layers.Reshape(target_shape=(config.hidden_dim,))

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: [(None,hiddendim),[(None,hidden_dim),...,(None,hidden_dim)]]
        #           =[(1,64),[(1,64),...,(1,64)]]
        # mask: (None,1,n)=(1,1,15), bool,  n=15: max_num_agents

        attention_mask = tf.cast(mask, 'bool')  # (None,1,n)=(1,1,15)

        query_feature = self.query_feature(inputs[0])  # (None,1,hidden_dim)=(1,1,64)
        features = self.features(inputs[1])  # (Nonen,n,hidden_dim)=(1,15,64)

        x, score = \
            self.mha1(
                query=query_feature,
                key=features,
                value=features,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,1,hidden_dim),(None,num_heads,1,n)=(1,1,64),(1,2,1,15)

        x1 = self.add1([inputs[0], x])  # (None,1,hidden_dim)=(1,1,64)

        x2 = self.dense(x1)  # (None,n,hidden_dim)=(1,1,64)

        x2 = self.dropoout1(x2, training=training)

        feature = self.add2([x1, x2])  # (None,1,hidden_dim)=(1,1,64)

        feature = self.reshape(feature)  # (1,64)

        batch_dim = inputs[0].shape[0]
        score = tf.reshape(score,
                           shape=(batch_dim, self.config.num_heads, self.config.max_num_red_agents)
                           )  # (1,2,15)

        return feature, score  # (None,hidden_dim), (None,num_heads,n)


class QLogitModel(tf.keras.models.Model):
    """
    Very simple dense model, output is logits

    :param action_dim=5
    :param hidden_dim=64
    :return: (None,action_dim)=(None,5)
    """

    def __init__(self, config, **kwargs):
        super(QLogitModel, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim * 3,
                activation='relu',
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.Dense(
                units=self.config.hidden_dim,
                activation='relu',
            )

        self.dense3 = \
            tf.keras.layers.Dense(
                units=self.config.action_dim,
                activation=None,
            )

    @tf.function
    def call(self, inputs, training=True):
        # inputs: (None,hidden_dim)=(None,4)

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)=(1,192)

        x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,hidden_dim)=(1,64)

        logit = self.dense3(x1)  # (None,action_dim)=(1,5)

        return logit  # (None,action_dim)=(1,5)


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

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

    """ cnn_model """
    cnn = CNNModel(config=config)

    # Get features list of all agents
    features = []

    for state in padded_states:
        feat = cnn(state)  # (1,64)
        features.append(feat)  # [(1,64),...,(1,64)], len=15

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    # Get output list of attention of all agents
    att_features = []
    att_scores = []
    for i in range(max_num_agents):
        query_feature = features[i]  # (1,64)

        inputs = [query_feature, features]  # [(1,64),[(1,64),...,(1,64)]]

        att_feature, att_score = \
            mha(inputs,
                masks[i],
                training=True
                )  # (None,hidden_dim),(None,num_heads,n)

        att_features.append(att_feature)
        att_scores.append(att_score)

    """ q_model """
    q_net = QLogitModel(config=config)

    # Get q_logits list of all agents
    q_logits = []
    for i in range(max_num_agents):
        q_logit = q_net(att_features[i])  # (None,5)
        q_logits.append(q_logit)

    pass


if __name__ == '__main__':
    main()
