import pandas as pd
import os


def create_outputs_directory():
    output_directory = "out"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def save_dicts(filename, dicos, cols, index):
    df = pd.DataFrame(columns=cols)
    for dico in dicos:
        df = df.append(dico, ignore_index=True)

    df.index = index
    df.to_csv("out/" + filename)


def save_dicts_perms(filename, dicos, index):
    df = pd.DataFrame(dicos)
    df.index = index
    df.to_csv("out/" + filename)
