# -*- coding: utf-8 -*-

"""Process the data

Padding, Delete, Replace and so on
"""
import numpy
import numpy as np
from gensim.models import Word2Vec
from getdata import get_data
# from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


def ethnicity2int(eth):
    """Encoding the ethnicity to int

    :param eth: the ethnicity need to encoder

    :return: code
    """

    Eth = {
        '': 0,
        'Caucasian': 1,
        'Hispanic': 2,
        'Asian': 3,
        'African American': 4,
        'Native American': 5,
        'Other/Unknown': 6,
    }
    if eth not in Eth.keys():
        return -1
    if not eth:
        return -1
    return float(Eth[eth])


def data_processing(data, data_type):
    """

    :param data:
    :param data_type:
    :return:
    """

    Data = []  # orig data
    Data_new = []  # processed data
    example = []
    drop = []  # dropping data

    for row in data:
        for attribute in row:
            if not attribute:
                example.append(None)
            else:
                example.append(attribute)
        Data.append(example)

        if None in example:  # If example exists None value, Drop the data
            drop.append(example)
        else:
            Data_new.append(example)
        example = []

    if data_type == 'Patient':  # Process the Patient basic information data
        for row in Data_new:
            for index, attribute in enumerate(row):
                if index == 0:
                    continue

                elif index == 1:
                    if attribute == 'Felmale':
                        row[index] = int(0)
                    else:
                        row[index] = int(1)

                elif index == 2:
                    if '>' in attribute:
                        row[index] = int(90)
                    else:
                        row[index] = int(data)

                elif index == 3:
                    row[index] = ethnicity2int(attribute)

                else:
                    row[index] = float(attribute)

    elif data_type == 'pastHistory':
        return Data_new

    elif data_type == 'Diagnosis':
        return Data_new

    elif data_type == 'Lab':
        for row in Data_new:
            for index, attribute in enumerate(row):
                if index == 3:
                    row[index] = int(data)

    else:
        return Data_new

    return Data_new


def train(sentences):
    """

    :param sentences:
    :return:
    """
    model = Word2Vec(sentences,
                     vector_size=10,
                     sg=1,
                     window=6,
                     min_count=1)
    model.save('model/model1.model')

    return model


def construct_diagnosis_vec_by_pudmed(diagnosis_dic):
    """

    :param diagnosis_dic:
    :return:
    """

    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    diagnosis_vec = {}

    with open(r"X:\4_Project\EHR_new\model\model.txt", "w") as f:
        for index, diagnosis in enumerate(diagnosis_dic):
            # print(model.wv[diagnosis[0]])
            embeddings = model.encode(diagnosis)
            embeddings = TSNE(n_components=3, perplexity=1).fit_transform(embeddings)
            vector = embeddings[0]
            for index, embedding in enumerate(embeddings):
                if index == 0:
                    continue
                vector += embedding
            diagnosis_name = diagnosis[0]

            for index, layer in enumerate(diagnosis):
                if index != 0:
                    diagnosis_name = diagnosis_name + "|" + layer

            diagnosis_vec[diagnosis_name] = vector / len(diagnosis)
            print(diagnosis_vec[diagnosis_name])
            f.write(diagnosis_name)
            for vec in diagnosis_vec[diagnosis_name]:
                f.write("~" + str(vec))
            f.write("\n")


def read_model_txt():
    """

    :return:
    """
    diagnosis_vec = {}
    with open(r"D:\Code\Python\EHR_new\model\model.txt", "r") as F:
        lines = F.readlines()
        for line in lines:
            sentences = line.split("~")
            vec = []
            for index, sentence in enumerate(sentences):
                if index == 0:
                    continue
                vec.append(float(sentence))
            diagnosis_vec[sentences[0]] = vec

    return diagnosis_vec


def data_padding_diagnosis(modname='word2vec'):
    """

    :return:
    """
    diagnosis_ = [row[2] for row in get_data(data_type='Diagnosis')]
    diagnosis_dic = []
    diagnosis_dic_ = []

    for diagnosis in diagnosis_:
        # context = ""
        context = []

        for sub_diagnosis in diagnosis.split("|"):
            # context += sub_diagnosis + ' '
            context.append(sub_diagnosis)

        if diagnosis not in diagnosis_dic_:
            diagnosis_dic_.append(diagnosis)
        if context not in diagnosis_dic:
            diagnosis_dic.append(context)

    if modname == 'word2vec':
        train(diagnosis_dic)
        diagnosis_vec_dic = construct_diagnosis_vec(diagnosis_dic)
        return diagnosis_vec_dic
    elif modname == 'pudmed2vec':
        # construct_diagnosis_vec_by_pudmed(diagnosis_dic)
        return read_model_txt()


def construct_diagnosis_vec(diagnosis_dic):
    """

    :return:
    """

    model = load_model()
    diagnosis_vec = {}
    for diagnosis in diagnosis_dic:
        # print(model.wv[diagnosis[0]])
        diagnosis_name = diagnosis[0]
        vector = np.fromstring(model.wv[diagnosis[0]], dtype=np.float32)
        # print(vector)

        for index, layer in enumerate(diagnosis):
            if index != 0:
                diagnosis_name = diagnosis_name + "|" + layer
                vector += np.fromstring(model.wv[layer], dtype=np.float32)
        diagnosis_vec[diagnosis_name] = vector / len(diagnosis)

    return diagnosis_vec


def load_model(model_path='model/model1.model'):
    """

    :param model_path:
    :return:
    """
    return Word2Vec.load(model_path)


if __name__ == '__main__':
    print(read_model_txt())
