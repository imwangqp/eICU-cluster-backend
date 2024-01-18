# -*- coding: utf-8 -*-
"""Get Data from icu patient dataset

There are some function about the data preprocessing and data getting
"""

import os
import sqlite3
import math
import json
import numpy as np
import functools

# the absolute path about
file = os.path.abspath('Data/identifier.sqlite')
dataset1 = sqlite3.connect(file, check_same_thread=False)


def get_sql(data_type):
    """Get the correct SQL

    :param Datatype: the data type which you want to get from dataset

    :return: A string type which is about the detailed sql
    """
    SQL = {
        'Patient': 'Select patientunitstayid, '
                   'gender, '
                   'age, '
                   'ethnicity, '
                   'admissionheight, '
                   'admissionweight '
                   'From patient;',

        'pastHistory': 'Select pasthistoryid, '
                       'patientunitstayid, '
                       'pasthistoryoffset, '
                       'pasthistoryvaluetext '
                       'From pastHistory;',

        'Diagnosis': 'Select diagnosisid, '
                     'patientunitstayid, '
                     'diagnosisstring, '
                     'diagnosispriority '
                     'From diagnosis',

        'Lab': 'Select labid, '
               'patientunitstayid, '
               'labname, '
               'labresulttext '
               'From lab',

        'diagnosis': 'Select diagnosisid,'
                     'diagnosisoffset,'
                     'diagnosisstring,'
                     'activeupondischarge '
                     'From diagnosis',

        'treatment': 'Select treatmentid,'
                     'treatmentoffset, '
                     'treatmentstring,'
                     'activeupondischarge '
                     'From treatment',

        'past_history': 'Select * '
                        'From pastHistory',

        'lab': 'Select * '
               'From lab',
    }
    return SQL[data_type]


def open_db(db_path=file):
    return sqlite3.connect(db_path, check_same_thread=False)


def get_data(dataset=open_db(), data_type=None):
    """Get data form dataset

    :param dataset:
    :param data_type: the data type witch user want to get

    :return: A list about the getting data from dataset
    """

    sql = get_sql(data_type)
    cursor = dataset.execute(sql)
    data = cursor.fetchall()

    return data


def get_info_by_id(dataset, table, attributes, patient_id):
    """Get the information which users want to get by patient id from dataset

    :param attributes: the query attribute from the table
    :param dataset: dataset
    :param table: query table
    :param patient_id: query patient_id
    :return: the tuple list about query records
    """

    sql = "Select "
    for attribute in attributes:
        if attribute == attributes[-1]:
            sql += attribute + ' '
        else:
            sql += attribute + ', '
    sql += ('From ' + table + ' where patientunitstayid == ' + patient_id)

    cursor = dataset.execute(sql)
    info = cursor.fetchall()
    return info


@functools.cache
def get_patient_history(patient_id, patient_history):
    """Get the patient disease history from dataset

    :param patient_id: Patient unique id
    :param patient_history: All history records about patient's disease

    :return: A list about the selected patient's history with temporal order
    """

    select_history = []
    patient_id_history = [row[1] for row in patient_history]

    if patient_id not in patient_id_history:
        return 0

    for history_data in enumerate(patient_history):
        if patient_id == history_data[1]:
            select_history.append(history_data)

    select_history.sort(key=lambda x: x[2])

    return select_history


def time_sort(time):
    return (min(time),
            max(time),
            sorted(range(len(time)), key=lambda index: time[index]))


@functools.cache
def get_detail_by_id(patient_id, vitalPeriodic_choice, time_block_len=120):
    """Get single patient's all detail from table

    :param time_block_len:
    :param patient_id: patient id
    :param vitalPeriodic_choice: the metric want to know
    :param Time_block_len: time block length
    :return: A json structure
    """

    vitalPeriodic_choices = {
        1: "temperature",
        2: "sao2",
        3: "heartrate",
        4: "respiration",
        5: "cvp",
        6: "etco2",
        7: "systemicsystolic",
        8: "systemicdiastolic",
        9: "systemicmean",
        10: "pasystolic",
        11: "padiastolic",
        12: "pamean",
        13: "st1",
        14: "st2",
        15: "st3",
        16: "icp"
    }
    tables = ['vitalPeriodic',
              'diagnosis',
              'treatment',
              'medication',
              'infusiondrug',
              'lab'
              ]
    attributes = {
        'vitalPeriodic': ["observationoffset",
                          vitalPeriodic_choices[vitalPeriodic_choice], ],
        'diagnosis': ["diagnosisoffset",
                      "activeupondischarge",
                      "diagnosispriority",
                      "diagnosisstring", ],
        'treatment': ["treatmentoffset",
                      "activeupondischarge",
                      "treatmentstring", ],
        'medication': ["drugstartoffset",
                       "drugstopoffset",
                       "drugname",
                       "dosage",
                       "routeadmin",
                       "frequency", ],
        'infusiondrug': ["infusionoffset",
                         "drugname",
                         "drugrate",
                         "infusionrate",
                         "drugamount",
                         "volumeoffluid",
                         "patientweight", ],
        'lab': ["labresultoffset",
                "labname",
                "labresult",
                "labmeasurenameinterface", ],
    }
    dataset = open_db()

    all_info = []
    for table in tables:
        info = get_info_by_id(dataset, table, attributes[table], patient_id)
        for instance in info:
            tab = (table,)
            if None not in instance:
                all_info.append(tab.__add__(instance))
    time = []
    for info in all_info:
        time.append(info[1])

    min_time, max_time, time_sort_index = time_sort(time)
    sorted_info = [all_info[i] for i in time_sort_index]

    detail = {"patientunitid": patient_id,
              "intervalCount": math.ceil((max_time - min_time) / time_block_len)}
    time_blocks = []
    for block_index in range((detail["intervalCount"])):
        time_block = {
            "time_block_index": block_index + 1,  # from 1 to n
            "context": {}
        }
        if block_index == detail["intervalCount"]:
            time_block["fin"] = 1
        else:
            time_block["fin"] = 0
        if block_index * time_block_len + min_time <= 0 < (block_index + 1) * time_block_len + min_time:
            time_block["enter_icu"] = 1
        else:
            time_block["enter_icu"] = 0
        for table in tables:
            time_block["context"][table] = []
        time_blocks.append(time_block)

    for context in sorted_info:
        attribute = {}
        table = context[0]
        offset = context[1] - min_time
        for index, cont in enumerate(context):
            if index > 1:
                attribute[attributes[table][index - 1]] = cont
            elif index == 1:
                attribute["offset"] = offset
            else:
                continue
        if table == "medication":
            for index in range(int(offset / time_block_len), int(int(context[2]) / time_block_len)):
                if index < detail["intervalCount"]:
                    time_blocks[index]["context"][table].append(attribute)
        else:
            time_blocks[int(offset / time_block_len)]["context"][table].append(attribute)

    # for time_block_index, time_block_ in enumerate(time_blocks):
    #     vitalPeriodices = time_block_['context']['vitalPeriodic']
    #     vitalperiodices = {}
    #     if vitalPeriodices:
    #         vita = [[], [], [], [], [], []]
    #         vita_mean = []
    #         vita_max = []
    #         vita_min = []
    #         len = int(time_block_len / 5)
    #         base_time = time_block_index * time_block_len
    #
    #         for vitalPeriodice in vitalPeriodices:
    #             vita[int((vitalPeriodice['offset'] - base_time) / len)].append(
    #                 vitalPeriodice[vitalPeriodic_choices[vitalPeriodic_choice]])
    #         for a in vita:
    #             if a:
    #                 vita_mean.append(np.mean(a))
    #                 vita_min.append(min(a))
    #                 vita_max.append(max(a))
    #
    #         vitalperiodices['offset'] = vitalPeriodices[0]['offset']
    #         vitalperiodices['min'] = min(vita_min)
    #         vitalperiodices['max'] = max(vita_max)
    #         vitalperiodices['value'] = vita_mean
    #         time_block_['context']['vitalPeriodic'] = vitalperiodices

    detail["time_block"] = time_blocks
    return json.loads(json.dumps(detail, ensure_ascii=False))


@functools.lru_cache()
def get_patient_by_id(patient_id):
    """

    :param patients_id:
    :return:
    """

    dataset = open_db()
    tables = ['patient', 'apachePatientResult', 'diagnosis', 'treatment', 'pastHistory', ]  # 'admissiondrug'
    attributes = {
        'patient': ['unittype',
                    'ethnicity',
                    'unitadmittime24',
                    'unitdischargetime24',
                    'admissionweight',
                    'admissionheight',
                    ],
        'apachePatientResult': [['apachescore'],
                                ['predictedhospitalmortality',
                                 'actualhospitalmortality',
                                 'predictedicumortality',
                                 'actualicumortality'],
                                ['predictedhospitallos',
                                 'actualhospitallos',
                                 'predictediculos',
                                 'actualiculos'],
                                ],
        'diagnosis': ['diagnosisid',
                      'diagnosisoffset',
                      'diagnosisstring',
                      'activeupondischarge'
                      ],
        'treatment': ['treatmentid',
                      'treatmentoffset',
                      'treatmentstring',
                      'activeupondischarge'
                      ],
        'pastHistory': ['pasthistoryid',
                        'pasthistoryoffset',
                        'pasthistoryvalue']
    }

    patient_info = {'id': patient_id}
    for table in tables:

        if table == 'patient':
            info_age = get_info_by_id(dataset, table, ['age'], patient_id)
            # print(info_age)
            if '>' in info_age[0][0]:
                patient_info['age'] = 90
            elif info_age[0][0] is None or info_age[0][0] == '':
                patient_info['age'] = int(0)
            else:
                patient_info['age'] = int(info_age[0][0])
            patient_info['gender'] = get_info_by_id(dataset, table, ['gender'], patient_id)[0][0]
            HW = get_info_by_id(dataset, table, ['admissionweight', 'admissionheight'], patient_id)
            if len(HW[0]) != 2:
                patient_info['BMI'] = round(HW[0][0] / math.pow(HW[0][1] / 100, 2), 2)
            else:
                patient_info['BMI'] = 0
            patient_info['unitdischargeoffset'] = round(
                (float(get_info_by_id(dataset, table, ['unitdischargeoffset'], patient_id)[0][0]) /
                 float(60 * 24)), 2)
            attribute = attributes[table]
            info = get_info_by_id(dataset, table, attribute, patient_id)
            for i, att in enumerate(attribute):
                if info[0][i] is None:
                    patient_info[att] = 0
                else:
                    patient_info[att] = info[0][i]

        elif table == 'apachePatientResult':
            for i, attribute in enumerate(attributes['apachePatientResult']):
                if i == 0:
                    score = get_info_by_id(dataset, table, attribute, patient_id)
                    if score:
                        patient_info['apachescore'] = score[0][0]
                    else:
                        patient_info['apachescore'] = 0
                else:
                    value = {}
                    info = get_info_by_id(dataset, table, attribute, patient_id)
                    if info:
                        for index, att in enumerate(info[0]):
                            value[attribute[index]] = att
                    else:
                        for index in range(4):
                            value[attribute[index]] = 0
                    if i == 1:
                        patient_info['morality'] = value
                    else:
                        patient_info['LOS'] = value

        else:
            attribute = attributes[table]
            info = get_info_by_id(dataset, table, attribute, patient_id)
            context = []
            if info:
                for info_i in info:
                    Context = {}
                    for index, att in enumerate(info_i):
                        if att is None:
                            Context[attribute[index]] = 0
                        else:
                            Context[attribute[index]] = att
                    context.append(Context)
                patient_info[table] = context

    return patient_info


@functools.cache
def fetch_all_detail(detail_id, time_block_index, all_detail, patient_id):
    """

    :param patient_id:
    :param all_detail:
    :param detail_id:
    :param time_block_index:
    :return:
    """

    details = ['diagnosis',
               'treatment',
               'medication',
               'infusionDrug',
               'lab',
               'temperature',
               'sao2',
               'heartrate',
               'respiration',
               'cvp',
               'etco2',
               'systemicsystolic',
               'systemicdiastolic',
               'systemicmean',
               'pasystolic',
               'padiastolic',
               'pamean',
               'st1', 'st2', 'st3',
               'icp',
               ]
    sub_detail = all_detail["time_block"][time_block_index - 1]
    if detail_id <= 4:
        detail = {"detail": sub_detail["context"][details[detail_id]],
                  "offset": (sub_detail["time_block_index"] - 1) * 120}
    else:
        detail = {"offset": (sub_detail["time_block_index"] - 1) * 120}
        sql = "Select " + details[
            detail_id] + ",observationoffset From vitalPeriodic where patientunitstayid=" + patient_id
        dataset = open_db()
        data = dataset.execute(sql).fetchall()
        detail_list = []
        for sample in data:
            detail_list.append({"offset": sample[1], "value": sample[0]})
        detail["detail"] = detail_list
    return json.loads(json.dumps(detail, ensure_ascii=False))


@functools.cache
def get_cluster_data():
    """

    :return:
    """
    sql = ("Select patient.patientunitstayid,"
           "patient.gender,"
           "patient.age,"
           "patient.ethnicity,"
           "patient.admissionheight,"
           "patient.admissionweight,"
           "diagnosis.diagnosisstring "
           "From patient "
           "INNER JOIN diagnosis "
           "ON patient.patientunitstayid = diagnosis.patientunitstayid"
           )

    dataset = open_db()
    cursor = dataset.execute(sql)
    cluster_data = cursor.fetchall()
    return cluster_data


def remove_duplicate_lists(lst):
    """

    :param lst:
    :return:
    """
    return [list(t) for t in set(map(tuple, lst))]


@functools.cache
def cluster_tree_map(patients_id):
    """

    :param patients_id:
    :return:
    """
    patients_id = [str(i) for i in patients_id]
    dataset = open_db()
    treemap = []
    data_new = []
    tree = {}
    for patient_id in patients_id:
        data = dataset.execute("Select diagnosisstring from diagnosis where patientunitstayid=" + patient_id)
        diagnosis_ = [row[0] for row in data]

        for sample in diagnosis_:
            data_new.append(sample.split('|'))
        for samples in data_new:
            flag = tree
            for i, sample in enumerate(samples):
                if sample not in flag.keys():
                    flag[sample] = {"value": 1}
                    flag = flag[sample]
                else:
                    flag[sample]["value"] += 1
                    flag = flag[sample]
    for subtree in tree:
        treemap.append(build_tree(tree[subtree], subtree))

    return treemap


def build_tree(dict, dictname):
    """

    :param dict:
    :param dictname:
    :return:
    """
    if len(dict) == 1:
        return {"name": dictname, "value": dict["value"]}
    else:
        return {"name": dictname,
                "value": dict["value"],
                "children": [build_tree(dict[sub_dictname],
                                        sub_dictname) for sub_dictname in dict if sub_dictname != "value"]}


def get_patients_by_id(patients_id):
    """

    :param patients_id:
    :return:
    """
    patients_info = []
    for patient_id in patients_id:
        patients_info.append(get_patient_by_id(patient_id))
    return json.loads(json.dumps({"patient": patients_info}, ensure_ascii=False))


@functools.cache
def fetch_all_detail(patient_id):
    """

    :param patient_id:
    :param all_detail:
    :param detail_id:
    :param time_block_index:
    :return:
    """
    table = {
        'diagnosis': ["diagnosisoffset",
                      "activeupondischarge",
                      "diagnosispriority",
                      "diagnosisstring", ],
        'treatment': ["treatmentoffset",
                      "activeupondischarge",
                      "treatmentstring", ],
        'medication': ["drugstartoffset",
                       "drugstopoffset",
                       "drugname",
                       "dosage",
                       "routeadmin",
                       "frequency", ],
        'infusiondrug': ["infusionoffset",
                         "drugname",
                         "drugrate",
                         "infusionrate",
                         "drugamount",
                         "volumeoffluid",
                         "patientweight", ],
        'lab': ["labresultoffset",
                "labname",
                "labresult",
                "labmeasurenameinterface", ],
    }
    details = ['diagnosis',
               'treatment',
               'medication',
               'infusiondrug',
               'lab',
               'temperature',
               'sao2',
               'heartrate',
               'respiration',
               'cvp',
               'etco2',
               'systemicsystolic',
               'systemicdiastolic',
               'systemicmean',
               'pasystolic',
               'padiastolic',
               'pamean',
               'st1', 'st2', 'st3',
               'icp',
               ]
    all_detail = {}
    db = open_db()
    for index, detail in enumerate(details):
        if index < 5:
            detail_list = []
            info = get_info_by_id(db, detail, table[detail], patient_id)
            for example in info:
                detail_dict = {}
                for a_index, att in enumerate(table[detail]):
                    if a_index == 0:
                        detail_dict["offset"] = example[a_index]
                    else:
                        detail_dict[att] = example[a_index]
                detail_list.append(detail_dict)
            all_detail[detail] = detail_list
        else:
            detail_list = []
            info = get_info_by_id(db, 'vitalPeriodic', ["observationoffset", detail], patient_id)
            for example in info:
                if example[1]:
                    detail_dict = {"offset": example[0],
                                   detail: example[1]}
                    detail_list.append(detail_dict)
            if detail_list:
                all_detail[detail] = detail_list

    return json.loads(json.dumps(all_detail, ensure_ascii=False))


if __name__ == "__main__":
    print(get_patients_by_id(["141764"]))
