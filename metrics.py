import numpy as np
import pandas as pd
from savingOutputs import save_dicts


def compute_metric(filename, subject_ids, metric, mode, ret = False):
    confusion_matrixes = pd.read_csv(filename+"confusion_matrixes_"+mode+".csv", index_col=0)
    score = [dict() for _ in subject_ids] 
    for modality in confusion_matrixes:
        for i, subj_id in enumerate(subject_ids):
            cf = confusion_matrixes[modality][subj_id].replace("[","").replace("]","").replace("\n","")
            if mode == "within" : cf = cf.split('.')[:-1]
            else : cf = [i for i in cf.split(' ') if i != '']
            cf = np.asarray(list(map(int, cf))).reshape(4,4)
            score[i][modality] = metric["function"](cf)
    save_dicts(filename + metric["name"] +"_" + mode + ".csv", score, list(score[0].keys()), subject_ids)
    if ret : return score


def accuracy(confusion_matrix):
    sum_Tis = sum([confusion_matrix[i][i] for i in range(4)])
    return sum_Tis/np.sum(confusion_matrix)


def recall(confusion_matrix):
    recall = [0]*4
    for i in range(4):
        recall[i] = confusion_matrix[i][i]/sum([confusion_matrix[j][i] for j in range(4)])
    return recall


def precision(confusion_matrix):
    precision = [0]*4
    for i in range(4):
        precision[i] = confusion_matrix[i][i]/sum(confusion_matrix[i])
    return precision