# -*- coding: utf-8 -*-
"""The Class of Cluster

the module describe the different method and afford methods to clustering the data
"""
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from process import data_padding_diagnosis
from process import ethnicity2int
from getdata import get_cluster_data
from getdata import cluster_tree_map
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn_extra.cluster import KMedoids
from sklearn_extra.cluster import CLARA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from kmodes.kprototypes import KPrototypes

cluster_patients_id = []


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def sum_float(arr1, arr2):
    sum = []
    for i in range(len(arr1)):
        add = arr1[i] + arr2[i]
        if math.isinf(add):
            sum.append(0)
            continue
        sum.append(add)

    return sum


def deduce_dimension(data, dimension=2, method='pca'):
    """

    :param dimension:  target dimension(default 2)
    :param data: source data
    :param method: deduce_dimension method: pac, tsne
    :return: target dimension list
    """

    if method == 'pca':
        model = PCA(n_components=dimension, whiten=True, svd_solver='randomized')
        model.fit(data)
        new_data = model.fit_transform(data)
    elif method == 'tsne':
        tsne = TSNE(n_components=dimension, perplexity=75)
        new_data = tsne.fit_transform(data)
    else:
        raise NameError

    x_min, x_max = new_data.min(0), new_data.max(0)
    data_norm = (new_data - x_min) / (x_max - x_min)

    return data_norm


def cluster_methods(data, K=5, method=0, cate=None):
    """

    :param cate:
    :param data:
    :param K:
    :param method:
    :return:
    """

    if method == 0:
        return kmeans(data, K)
    elif method == 1:
        return kprototypes(data, K, cate)
    elif method == 2:
        return kmedoids(data, K)
    elif method == 3:
        return affinity_propagation(data, K)
    elif method == 4:
        return clara(data, K)
    elif method == 5:
        return dbscan(data)
    else:
        raise NameError


def data_process(cluster_data, diagnosis_dic, cluster_factor=None):
    """

    :param cluster_factor: [1, 0, 0, 0, 0, 1]
    :param diagnosis_dic:
    :param cluster_data:
    :return:
    """

    patients = []
    exist_patients = []

    for patient_info in cluster_data:
        patient = []
        if patient_info[0] not in exist_patients:
            exist_patients.append(patient_info[0])
            for index, info in enumerate(patient_info):
                if index < 4:
                    patient.append(info)
                elif index == 4:
                    continue
                elif index == 5:
                    if patient_info[4] and info:
                        patient.append(patient_info[4] / math.pow(info, 2))
                    else:
                        patient.append(-1)
                else:
                    patient.append(diagnosis_dic[info])
        else:
            patient_index = exist_patients.index(patient_info[0])
            patients[patient_index][5] = sum_float(patients[patient_index][5], diagnosis_dic[patient_info[6]])
        if patient:
            patients.append(patient)
    final_patients = []
    for patient in patients:
        final_patient = []
        for index, attributes in enumerate(patient):
            if cluster_factor[index] == 1:
                if index == 0:
                    final_patient.append(int(attributes))
                elif index == 2:
                    if '>' in attributes:
                        final_patient.append(float(90))
                    elif attributes == "":
                        final_patient.append(float(-1))
                    else:
                        final_patient.append(float(attributes))
                elif index == 1:
                    if attributes == '':
                        final_patient.append(0)
                    elif attributes == '':
                        final_patient.append(1)
                    else:
                        final_patient.append(-1)
                elif index == 3:
                    eth = ethnicity2int(attributes)
                    if not eth:
                        final_patient.append(eth)
                elif index == 5:
                    for attribute in attributes:
                        final_patient.append(attribute)
                else:
                    final_patient.append(attributes)

        if None not in final_patient:
            final_patients.append(final_patient)

    final_patients = np.array(final_patients, dtype=float)

    return final_patients


def plot_(Data, label, fig_size=8):
    """

    :param fig_size:
    :param label:
    :param Data:
    :return:
    """
    plt.figure(figsize=(fig_size, fig_size))
    for i in range(Data.shape[0]):
        plt.text(Data[i, 0], Data[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


def kmeans(X, K):
    """

    :param X:
    :param K:
    :return:
    """
    K = KMeans(n_clusters=K,
               max_iter=200,
               n_init=10,
               init='k-means++'
               )
    K.fit(X)
    labels = K.labels_
    center = K.cluster_centers_

    return ['Kmeans',
            labels,
            center,
            ]


def kmedoids(X, K):
    """

    :return:
    """
    K = KMedoids(n_clusters=K,
                 metric='euclidean',
                 method='pam',
                 init='k-medoids++',
                 max_iter=500,
                 random_state=None)
    K.fit(X)
    label = K.labels_
    center = K.cluster_centers_

    return ['KMedoids',
            label,
            center,
            ]


def dbscan(X):
    """

    :return:
    """
    db = DBSCAN(eps=100, min_samples=2)
    db.fit(X)
    labels = db.labels_

    n_cluster_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_special_ = list(labels).count(-1)

    return ['DBSCAN',
            labels,
            n_cluster_,
            n_special_,
            ]


def affinity_propagation(X, K):
    """

    :return:
    """
    af = AffinityPropagation(preference=-50, random_state=0)
    af.fit(X)
    labels = af.labels_
    n_clusters_ = len(af.cluster_centers_indices_)

    return ['Affinity_Propagation',
            labels,
            n_clusters_
            ]


def clara(X, K):
    """

    :return:
    """
    cl = CLARA(n_clusters=K, metric='euclidean', max_iter=500)
    cl.fit(X)
    labels = cl.labels_
    center = cl.cluster_centers_

    return ['CLARA',
            labels,
            center,
            ]


def kprototypes(X, K, cate):
    """

    :param X:
    :param K:
    :param cate:
    :return:
    """

    kpro = KPrototypes(n_clusters=K)
    kpro.fit(X, categorical=cate)
    label = kpro.labels_
    center = kpro.cluster_centroids_

    return ['kprototypes',
            label,
            center
            ]


def cluster(cluster_count, cluster_method=None, cluster_factor=None):
    """

    :param cluster_factor:
    :param tree_index:
    :param cluster_method:
    :param cluster_count:
    :return:
    """
    data_show = []
    cluster_result = {}
    point = []
    center = []
    similarity = []
    cluster_score = []
    cluster_data = get_cluster_data()
    diagnosis_dic = data_padding_diagnosis(modname="pudmed2vec")
    # diagnosis_dic = data_padding_diagnosis(modname='pudmed2vec')

    final_patients = data_process(cluster_data, diagnosis_dic, cluster_factor)

    data_norm = deduce_dimension(np.delete(final_patients, 0, axis=1), dimension=2, method='tsne')

    for count in range(2, 13):
        score = {"count": count}
        data_count = cluster_methods(data_norm, count, cluster_method)
        if count == cluster_count:
            data_show = data_count
        silhouette = silhouette_score(data_norm, data_count[1])
        db = davies_bouldin_score(data_norm, data_count[1])
        score["ch_score"] = silhouette
        score["db_score"] = db
        cluster_score.append(score)
    cluster_result["cluster_score"] = cluster_score

    sim_score = 1 - pairwise_distances(data_show[2], metric='euclidean')
    for x_index in range(cluster_count):
        center_ = {"id": x_index, "x": data_show[2][x_index][0], "y": data_show[2][x_index][1]}
        for y_index in range(x_index, cluster_count):
            similar_sample = {"x": x_index, "y": y_index, "similarity": sim_score[x_index][y_index]}
            similarity.append(similar_sample)
            if x_index != y_index:
                similar_sample1 = {"x": y_index, "y": x_index, "similarity": sim_score[x_index][y_index]}
                similarity.append(similar_sample1)
        center.append(center_)
    cluster_result["center"] = center
    cluster_result["similarity"] = similarity

    for index, patient in enumerate(final_patients):
        patient_ = {"id": patient[0].astype(int),
                    "x": data_norm[index][0],
                    "y": data_norm[index][1],
                    "cluster_id": data_show[1][index]}
        point.append(patient_)
    # plot_(data_norm, data_show[1])
    cluster_result["point"] = point

    global cluster_patients_id
    cluster_patients_id = []
    for i in range(cluster_count + 1):
        cluster_patients_id.append([])
    for index, id in enumerate(final_patients):
        cluster_patients_id[data_show[1][index]].append(str(int(id[0])))
        cluster_patients_id[cluster_count].append(str(int(id[0])))
    for index, cluster_patients in enumerate(cluster_patients_id):
        cluster_patients = tuple(cluster_patients)
        cluster_patients_id[index] = cluster_patients
    return json.loads(json.dumps(cluster_result, cls=NpEncoder, ensure_ascii=False))


def tree_map(cluster_count, cluster_index=-1):
    """

    :param cluster_count:
    :param cluster_index:
    :return:
    """

    if cluster_index != -1:
        treemap = {"treemap": cluster_tree_map(cluster_patients_id[cluster_index])}
    else:
        treemap = {"treemap": cluster_tree_map(cluster_patients_id[cluster_count])}
    return json.loads(json.dumps(treemap, cls=NpEncoder, ensure_ascii=False))


if __name__ == "__main__":
    print(cluster(10, 0, [1, 1, 1, 1, 1, 1]))
