
# -*- coding: utf-8 -*-
import getdata
from getdata import get_patients_by_id
from cluster import cluster
from cluster import tree_map
from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

patient_id = ""
detail = {}
cluster_count = 5


@app.route('/cluster/', methods=['POST'])
def cluster_():
    """

    :return:
    """
    global cluster_count

    json = request.json
    cluster_count = json["cluster_count"]
    cluster_factor = json["cluster_factor"]
    cluster_methods = json["cluster_method"]
    cluster_factors = [1]
    for index in range(5):
        if index in cluster_factor:
            cluster_factors.append(1)
        else:
            cluster_factors.append(0)
    return cluster(cluster_count, cluster_methods, cluster_factors)


@app.route('/fetch_detail_by_id/', methods=["POST"])
def fetch_detail_by_id():
    """
    :return:
    """
    global patient_id
    global detail

    json = request.json
    patient_id = json["patient_id"]
    vital = json['vitalPeriodic_choice']
    offset = json['offset_interval']
    detail = getdata.get_detail_by_id(patient_id, vital, offset)

    return detail


@app.route(rule='/fetch_patients_by_id', methods=["POST"])
def fetch_patients_by_id():
    """

    :return:
    """
    json = request.json
    patients_id = json["patients_id"]
    return get_patients_by_id(patients_id)


@app.route(rule='/fetch_all_detail', methods=["POST"])
def fetch_all_detail():
    """

    :return:
    """
    json = request.json
    tmp_patient_id = json["patient_id"]
    return getdata.fetch_all_detail(tmp_patient_id)


# @app.route(rule='/fetch_all_detail', methods=["POST"])
# def fetch_all_detail():
#     """
#
#     :return:
#     """
#     json = request.json
#     detail_id = json["detail_id"]
#     time_block_index = json["time_block_index"]
#     return getdata.fetch_all_detail(detail_id, time_block_index, detail, patient_id)


@app.route(rule='/tree_map/', methods=["POST"])
def fetch_treemap():
    """

    :return:
    """
    json = request.json
    cluster_index = json["cluster_index"]
    return tree_map(cluster_count, cluster_index)


if __name__ == '__main__':
    app.run(port=8001)
