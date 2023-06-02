import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro


def eva(data, y_true, y_pred, epoch=0, f="log.txt"):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    # nmi = np.round(nmi_score(y_true, y_pred), 5)
    ari = ari_score(y_true, y_pred)
    sc = silhouette_score(data, y_pred)
    # print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    print("MY epoch {}:acc {:.4f}, nmi {:.4f}, ari {:.4f}, f1 {:.4f}, sc {:.4f}".format(epoch, acc, nmi, ari, f1, sc))
    print("MY epoch {}:acc {:.4f}, nmi {:.4f}, ari {:.4f}, f1 {:.4f}, sc {:.4f}".format(epoch, acc, nmi, ari, f1, sc), file=f)
    return acc, nmi, ari, f1, sc

def cal_clustering_acc(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    return acc


def cal_clustering_metric(truth, prediction):
    acc = cal_clustering_acc(truth, prediction)
    nmi = metrics.normalized_mutual_info_score(truth, prediction)
    ari = metrics.adjusted_rand_score(truth, prediction)
    f1 = metrics.f1_score(truth, prediction, average='macro')
    return acc, nmi, ari, f1