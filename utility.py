import numpy as np
import time
import random

"""
File containing small utility functions
"""


def average_dicos(dicos):
    dico = dict()
    for key in dicos[0]:
        dico[key] = np.mean([dic[key] for dic in dicos])

    return dico


def compute_bootstrap_distribution(n_bootstrap, n_subjects, scores_n_perm, n_single_perm):
    start_time = time.time()

    scores_bootstrap = [None] * n_bootstrap
    for i in range(n_bootstrap):
        dicos = [dict() for _ in range(n_subjects)]
        for j in range(n_subjects):
            dicos[j] = scores_n_perm[j][random.randint(0, n_single_perm - 1)]
        scores_bootstrap[i] = average_dicos(dicos)

    duration = time.time() - start_time
    print("Running models done in " + str(duration) + " seconds")
    return scores_bootstrap