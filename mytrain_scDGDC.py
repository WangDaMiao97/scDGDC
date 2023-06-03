import random
import tensorflow.compat.v1 as tf
from utils import *
from time import time
import argparse
import matplotlib
matplotlib.use('TkAgg')
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from myModel_scDGDC import GAE
from evaluation import eva
from graph_function import *



# 输入参数设置
def get_args(dataset_name, dataset_path, dataset_label, log_path, model_pth, seed=0,
             pretrain_epochs=500, pretrain_alpha=0.1, maxiter=200, train_alpha=0.1, n_neighbors=15, n_pairs=0.1):
    # yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    parser = argparse.ArgumentParser(description='Parser for scDGDC')
    # Basic
    parser.add_argument("--seed", default=seed, type=int)
    parser.add_argument("--log_file", default=log_path,
                        type=str)  # log日志的存储路径
    # Dataset
    parser.add_argument("--dataset_name", default=dataset_name, type=str)
    parser.add_argument('--dataset_path', default=dataset_path, type=str,
                        help='path to dataset (adata)')
    parser.add_argument("--dataset_label", default=dataset_label, type=str)  # Sample label
    # Pretrain
    parser.add_argument("--pretrain_epochs", default=pretrain_epochs, type=int)  # Pre-trained epochs
    parser.add_argument("--pretrain_alpha", default=pretrain_alpha, type=float)  # Weight of triplet contrast loss during pre-training
    parser.add_argument("--n_neighbors", default=n_neighbors, type=int)  # Number of neighbors used in graph construction
    parser.add_argument("--n_pairs", default=n_pairs, type=float)  # Number of triplets
    parser.add_argument("--model_pth", default=model_pth, type=str)
    # Train
    parser.add_argument("--maxiter", default=maxiter, type=int)  # Trained epochs
    parser.add_argument("--train_alpha", default=train_alpha, type=float)

    args = parser.parse_args()
    return args


# Compute cluster centroids, which is the mean of all points in one cluster.
def computeMeanCentroids(data, DM_labels, KNN_labels):
    D = max(DM_labels.max(), KNN_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(KNN_labels.size):
        w[DM_labels[i], KNN_labels[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return np.array([data[np.logical_and(DM_labels == ind[i,0], KNN_labels == ind[i,1])].mean(0) for i in range(D)])

def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    print(y_pred.size)
    print(y_true.size)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == "__main__":

    dataset_name = "Qx_Mammary_Glan"
    dataset_path = "data/Qx_Mammary_Glan/Qx_Mammary_Glan_preprocessed.h5ad"
    dataset_label = "Group"
    model_pth = "data/Qx_Mammary_Glan/"
    log_path = "data/Qx_Mammary_Glan/Qx_Mammary_Glan_preprocessed.txt"
    args = get_args(dataset_name, dataset_path, dataset_label, log_path, model_pth, seed = 0)
    print(args)
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    random.seed(args.seed)


    # Load data
    f = open(args.log_file, 'w')
    adata = sc.read_h5ad(args.dataset_path)
    print(adata)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(adata.obs[args.dataset_label].values).astype("int32")  # yan.h5ad

    cluster_number = int(max(y) - min(y) + 1)
    print("cluster_number: ", cluster_number)

    count = adata.X
    hvg = adata.var['highly_variable']
    size_factor = adata.obs['size_factors']
    raw_count = adata.raw[:, hvg.index.values].X.todense()

    #  Graph construction
    DM_adj, DM_adj_n = get_adj_DM(adata, k=args.n_neighbors)
    KNN_adj, KNN_adj_n = get_adj(adata, k=args.n_neighbors)
    # Triplet contrast
    S = getS(count)
    if (args.n_pairs>0):
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(count, S, args.n_pairs)

    # Defining the model
    model = GAE(raw_count, count, size_factor, model_pth, DM_adj=DM_adj, DM_adj_n=DM_adj_n, KNN_adj=KNN_adj, KNN_adj_n=KNN_adj_n, S=S)

    # Pre-training
    t0 = time()
    if (args.n_pairs>0):
        model.pre_train(y, epochs=args.pretrain_epochs, ml_ind1=ml_ind1, ml_ind2=ml_ind2,
                        cl_ind2=cl_ind2, alpha=args.pretrain_alpha, f=f)
    else:
        model.pre_train(epochs=args.pretrain_epochs, f=f)
    print("Pretrain end!")
    t1 = time()
    print('Pretrain: run time is {:.2f} '.format(t1 - t0), 'minutes')
    print('Pretrain: run time is {:.2f} '.format(t1 - t0), 'minutes', file=f)

    # The initial centroids of clusters are obtained by Kmeans from the results of pre-training
    model.load_model("pretrain")
    X_pretrain = model.embedding(count)
    pca = PCA(n_components=15)
    countp = pca.fit_transform(count)
    labels = KMeans(n_clusters=cluster_number).fit(countp).labels_
    centers = computeCentroids(X_pretrain, labels)

    # Clustering training
    if (args.n_pairs>0):
        model.train(y, epochs=args.maxiter, centers=centers, ml_ind1=ml_ind1,
                    ml_ind2=ml_ind2, cl_ind2=cl_ind2, alpha=args.train_alpha, f=f)
    else:
        model.train(y, epochs=args.maxiter, centers=centers, f=f)

    if y is not None:
        model.load_model("train")
        X_train, y_pred = model.get_cluster()
        acc = np.round(cluster_acc(y, y_pred), 5)
        y = list(map(int, y))
        y_pred = np.array(y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('ACC= %.4f, NMI= %.4f, ARI= %.4f'
              % (acc, nmi, ari))

        print(X_train.shape)
        eva(X_train, y, y_pred, args.dataset_name + " " + str(args.n_neighbors) + " Finsh!", f=f)

