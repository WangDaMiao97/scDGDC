import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, Input, Lambda
from spektral.layers import TAGConv
from tensorflow.keras.initializers import GlorotUniform
from layers import *
from utils import  *
import numpy as np
import tensorflow_probability as tfp

class Single_Encoder(tf.keras.Model):
    def __init__(self, X, adj_n, hidden_dim=128, latent_dim=15):
        super(Single_Encoder, self).__init__()

        self.X = X
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.adj_n = adj_n
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        initializer = GlorotUniform()
        self.drop = Dropout(0.2)
        self.relu = Activation('relu')

        self.TAG_1 = TAGConv(channels=self.hidden_dim, kernel_initializer=initializer, name="TAG_1")
        self.bn = BatchNormalization()
        self.TAG_2 = TAGConv(channels=self.latent_dim, kernel_initializer=initializer, name="TAG_2")

    def call(self, inputs, training=False, **kwargs):

        x=inputs
        x_h = self.drop(x)

        h = self.TAG_1([x_h, self.adj_n])
        h = self.bn(h, training = training)
        h = self.relu(h)
        z = self.TAG_2([h, self.adj_n])

        return z

    def get_config(self):
        pass


class Dual_Encoder(tf.keras.Model):
    def __init__(self, X, DM_adj_n, KNN_adj_n, hidden_dim=128, latent_dim=15):
        super(Dual_Encoder, self).__init__()

        self.X = X
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.DM_adj_n = tfp.math.dense_to_sparse(DM_adj_n)
        self.KNN_adj_n = tfp.math.dense_to_sparse(KNN_adj_n)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        initializer = GlorotUniform()
        self.drop = Dropout(0.2)
        self.relu = Activation('relu')

        self.TAG_DM_1 = TAGConv(channels=self.hidden_dim, kernel_initializer=initializer, name="TAG_DM_1")
        self.bn_DM = BatchNormalization()
        self.TAG_DM_2 = TAGConv(channels=self.latent_dim, kernel_initializer=initializer, name="TAG_DM_2")
        self.TAG_KNN_1 = TAGConv(channels=self.hidden_dim, kernel_initializer=initializer, name="TAG_KNN_1")
        self.bn_KNN = BatchNormalization()
        self.TAG_KNN_2 = TAGConv(channels=self.latent_dim, kernel_initializer=initializer, name="TAG_KNN_2")
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):

        x=inputs
        x_h = self.drop(x)

        DM_h = self.TAG_DM_1([x_h, self.DM_adj_n])
        DM_h = self.bn_DM(DM_h, training = training)
        DM_h = self.relu(DM_h)
        DM_z = self.TAG_DM_2([DM_h, self.DM_adj_n])

        KNN_h = self.TAG_KNN_1([x_h, self.KNN_adj_n])
        KNN_h = self.bn_KNN(KNN_h, training = training)
        KNN_h = self.relu(KNN_h)
        KNN_z = self.TAG_KNN_2([KNN_h, self.KNN_adj_n])

        z = self.add([DM_z, KNN_z])

        return z

    def get_config(self):
        pass


class DecoderA(tf.keras.Model):
    def __init__(self, adj_dim=32):
        super(DecoderA, self).__init__()

        self.adj_dim = adj_dim

        self.Dense = Dense(units=self.adj_dim, activation=None)
        self.Bilinear = Bilinear()
        self.Lambda = Lambda(lambda z: tf.nn.sigmoid(z))

    def call(self, inputs, **kwargs):

        h = self.Dense(inputs)
        h = self.Bilinear(h)
        dec_out = self.Lambda(h)

        return dec_out

    def get_config(self):
        pass

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

class DecoderX(tf.keras.Model):
    def __init__(self, latent_dim, raw_dim, dec_dim=None):
        super(DecoderX, self).__init__()
        self.raw_dim = raw_dim

        if dec_dim is None:
            self.dec_dim = [128, 256, 512]
        else:
            self.dec_dim = dec_dim
        self.relu = Activation('relu')

        self.fc1 = Dense(units=self.dec_dim[0])
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(units=self.dec_dim[1])
        self.bn2 = BatchNormalization()
        self.fc3 = Dense(units=self.dec_dim[2])
        self.bn3 = BatchNormalization()

        self.fc_pi = Dense(units=self.raw_dim, activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')
        self.fc_disp = Dense(units=self.raw_dim, activation=DispAct, kernel_initializer='glorot_uniform', name='dispersion')
        self.fc_mean = Dense(units=self.raw_dim, activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')

    def call(self, inputs, training=False, **kwargs):

        x = inputs

        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x, training=training)
        x = self.relu(x)

        pi = self.fc_pi(x)
        disp = self.fc_disp(x)
        mean = self.fc_mean(x)

        return pi,disp,mean

    def get_config(self):
        pass
