import numpy as np
import pandas as pd
from saving_outputs import save_dicts
from utility import cfm_string_to_matrix


def compute_metric(filename, subjects_ids, metric, mode, masks_exist, n_classes, ret = False):
    confusion_matrixes = pd.read_csv(filename+"confusion_matrixes_"+mode+".csv", index_col=0)
    score = [dict() for _ in subjects_ids] 
    for modality in confusion_matrixes:
        for i, subj_id in enumerate(subjects_ids):
            spl = modality.split("_")
            region = spl[1]+"_"+spl[2]
            if masks_exist[i][region] :
                cf = cfm_string_to_matrix(confusion_matrixes[modality][subj_id])
                score[i][modality] = metric["function"](cf, n_classes)
    save_dicts(filename + metric["name"] +"_" + mode + ".csv", score, list(score[0].keys()), subjects_ids)
    if ret : return score


def compute_accuracy_bootstrap(n_subjects, n_single_perm, confusion_matrixes_bootstrap, n_classes):
    scores = [0]*n_subjects
    for i in range(n_subjects):
        scores[i] = [dict() for _ in range(n_single_perm)]
        for j in range(n_single_perm):
            cfm_bootstrap = confusion_matrixes_bootstrap[i][j]
            for modality in cfm_bootstrap:
                scores[i][j][modality] = accuracy(cfm_bootstrap[modality], n_classes)
    return scores


def compute_accuracy_variance(filename, mode):
    accuracies = pd.read_csv(filename+"accuracy_"+mode+".csv")
    var = accuracies.var()
    var.to_csv(filename+"var_"+mode+".csv")


def accuracy(confusion_matrix, n_classes):
    sum_Tis = sum([confusion_matrix[i][i] for i in range(n_classes)])
    return sum_Tis/np.sum(confusion_matrix)


def recall(confusion_matrix, n_classes):
    recall = [0]*n_classes
    for i in range(n_classes):
        recall[i] = confusion_matrix[i][i]/sum([confusion_matrix[j][i] for j in range(n_classes)])
    return recall


def precision(confusion_matrix, n_classes):
    precision = [0]*n_classes
    for i in range(n_classes):
        precision[i] = confusion_matrix[i][i]/sum(confusion_matrix[i])
    return precision