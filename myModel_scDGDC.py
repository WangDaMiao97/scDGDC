import tensorflow.keras.backend as K
from tensorflow.keras.losses import MSE, KLD
from module import *
import tensorflow_probability as tfp
import numpy as np
from sklearn import metrics
from loss import ZINB, pairwise_loss, cal_dist

# 定义梯度裁剪函数
def clip_gradients(gradients, clip_value):
    clipped_gradients = []
    for gradient in gradients:
        clipped_gradient = K.clip(gradient, -clip_value, clip_value)
        clipped_gradients.append(clipped_gradient)
    return clipped_gradients


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class GAE(tf.keras.Model):
    def __init__(self, raw_X, X, size_factor, model_pth, DM_adj, DM_adj_n, KNN_adj, KNN_adj_n, S, hidden_dim=128,
                 latent_dim=15, dec_dim=None, adj_dim=32):
        super(GAE, self).__init__()
        if dec_dim is None:
            dec_dim = [128, 256, 512]
            # dec_dim = [128, 256]
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.raw_X = raw_X
        self.X = X
        self.size_factor = tf.convert_to_tensor(size_factor, dtype="float32")
        self.model_pth = model_pth
        self.DM_adj = np.float32(DM_adj)
        self.DM_adj_n = tfp.math.dense_to_sparse(np.float32(DM_adj_n))
        self.KNN_adj = np.float32(KNN_adj)
        self.KNN_adj_n = tfp.math.dense_to_sparse(np.float32(KNN_adj_n))
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.raw_dim = raw_X.shape[1]
        self.sparse = True

        self.encoder = Dual_Encoder(X, DM_adj_n, KNN_adj_n,hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        self.decoderX = DecoderX(self.latent_dim, self.raw_dim)
        self.decoderA = DecoderA(adj_dim)

        self.cluster_layer = ClusteringLayer(name='clustering')
        print(self.cluster_layer)

        self.encoder.build(input_shape=(None, self.in_dim))
        self.decoderA.build(input_shape=(None, self.latent_dim))
        self.decoderX.build(input_shape=(None, self.latent_dim))
        self.cluster_layer.build(input_shape=(None, self.latent_dim))


    def pre_train(self, y, epochs=1000, info_step=20, lr=1e-3, W_a=0.0, W_x=1.0, alpha = 0.0, ml_ind1 = np.array([]),
                  ml_ind2 = np.array([]), cl_ind2 = np.array([]), f = "log.txt"):

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=lr, decay_steps=epochs, decay_rate=0.99)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        best_loss = 1e6
        count = 0
        # Pretraining
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder(self.X, training=True)
                pi, disp, mean = self.decoderX(z, training=True)
                zinb = ZINB(pi=pi, theta=disp, scale_factor=self.size_factor, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.raw_X, mean, mean=True)

                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.DM_adj, A_out)) + tf.reduce_mean(MSE(self.KNN_adj, A_out))

                if alpha>0:
                    z_norm = z
                    z_anchor = tf.gather(z_norm, axis=0, indices=np.array(ml_ind1))
                    z_pos = tf.gather(z_norm, axis=0, indices=np.array(ml_ind2))
                    z_neg = tf.gather(z_norm, axis=0, indices=np.array(cl_ind2))
                    reg_loss = pairwise_loss(z_anchor, z_pos, z_neg)
                    loss = W_a * A_rec_loss + W_x * zinb_loss + alpha * reg_loss
                else:
                    loss = W_a * A_rec_loss + W_x * zinb_loss
            # 定期输出损失值
            if epoch % info_step == 0:
                print("Epoch", epoch, "total_loss:", loss.numpy())

            # 早停准则
            if loss < best_loss:
                best_loss = loss
                count = 0
                self.encoder.save_weights(self.model_pth+"pretrain_encoder.h5")
                self.decoderX.save_weights(self.model_pth+"pretrain_decoderX.h5")
                self.decoderA.save_weights(self.model_pth+"pretrain_decoderA.h5")
            else:
                count += 1
            if count == 20:
                break

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            optimizer.apply_gradients(zip(grads, vars))

        print("Pre_train Finish!")
        print("Pre_train Finish!", f)

    def train(self, y, epochs=300, lr=5e-4, W_a=0.0, W_x=1.0, W_c=0.3, info_step=1, n_update=2, centers=None, alpha = 0.0,
              ml_ind1 = np.array([]), ml_ind2 = np.array([]), cl_ind2 = np.array([]), f = "log.txt"):

        print(self.cluster_layer.clusters)
        self.cluster_layer.clusters = centers
        print(self.cluster_layer.clusters)

        # Training
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=epochs, decay_rate=0.99)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        best_loss = 1e6
        y_last_pred = []
        for epoch in range(0, epochs):
            if epoch % n_update == 0:
                z = self.encoder(self.X)

                q = self.cluster_layer(z)
                p = self.target_distribution(q)
                y_pred = q.numpy().argmax(1)
                if epoch==0:
                    y_last_pred = y_pred
                acc = np.round(cluster_acc(y, y_pred), 5)

                y_pred = np.array(y_pred)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                if epoch == 0 or len(set(y_pred))==1:
                    sc = 0.0
                else:
                    sc = np.round(metrics.silhouette_score(z, y_pred), 5)
                print('epoch=%d, ACC= %.4f, NMI= %.4f, ARI= %.4f, SC=%.4f'
                      % (epoch, acc, nmi, ari, sc), file=f)

            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder(self.X, training=True)

                q_out = self.cluster_layer(z)
                cluster_loss = tf.reduce_mean(KLD(q_out, p))

                y_pred = q_out.numpy().argmax(1)
                delta_y = 1 - cluster_acc(y_last_pred, y_pred)
                y_last_pred = y_pred

                pi, disp, mean = self.decoderX(z, training = True)
                zinb = ZINB(pi, theta=disp, scale_factor=self.size_factor, ridge_lambda=0, debug=False)
                zinb_loss = zinb.loss(self.raw_X, mean, mean=True)

                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.DM_adj, A_out)) + tf.reduce_mean(MSE(self.KNN_adj, A_out))

                dist1, dist2 = cal_dist(z, self.cluster_layer.clusters)
                soft_kmeans = tf.reduce_mean(tf.reduce_sum(dist2, axis=1))

                if alpha > 0:
                    z_norm = z
                    z_anchor = tf.gather(z_norm, axis=0, indices=np.array(ml_ind1))
                    z_pos = tf.gather(z_norm, axis=0, indices=np.array(ml_ind2))
                    z_neg = tf.gather(z_norm, axis=0, indices=np.array(cl_ind2))
                    reg_loss = pairwise_loss(z_anchor, z_pos, z_neg)
                    tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss + 0.01*soft_kmeans + alpha * reg_loss
                else:
                    tot_loss = W_a * A_rec_loss + W_x * zinb_loss + W_c * cluster_loss + 0.01*soft_kmeans
                stop_loss = tot_loss

            if epoch % info_step == 0:
                print("Epoch", epoch, "total_loss:", tot_loss.numpy())

            # 早停准则
            if stop_loss < best_loss:
                best_loss = stop_loss
                loss_count = 0
                self.encoder.save_weights(self.model_pth + "train_encoder.h5")
                self.decoderX.save_weights(self.model_pth + "train_decoderX.h5")
                self.decoderA.save_weights(self.model_pth + "train_decoderA.h5")
            else:
                loss_count += 1


            if delta_y<0.001 or loss_count == 10:
                if delta_y<0.001:
                    self.encoder.save_weights(self.model_pth + "train_encoder.h5")
                    self.decoderX.save_weights(self.model_pth + "train_decoderX.h5")
                    self.decoderA.save_weights(self.model_pth + "train_decoderA.h5")
                break

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            optimizer.apply_gradients(zip(grads, vars))


    def embedding(self, count):
        embedding = self.encoder(count)
        return np.array(embedding)

    def get_cluster(self):
        z = self.encoder(self.X)
        q = self.cluster_layer(z)
        q = q.numpy()
        y_pred = q.argmax(1)
        return z, y_pred

    def target_distribution(self, q):
        q = q.numpy()
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def load_model(self, mode="pretrain"):
        if mode=="pretrain":
            self.encoder.load_weights(self.model_pth+"pretrain_encoder.h5")
            self.decoderA.load_weights(self.model_pth + "pretrain_decoderA.h5")
            self.decoderX.load_weights(self.model_pth + "pretrain_decoderX.h5")
        elif mode == "train":
            self.encoder.load_weights(self.model_pth + "train_encoder.h5")
            self.decoderA.load_weights(self.model_pth + "train_decoderA.h5")
            self.decoderX.load_weights(self.model_pth + "train_decoderX.h5")

