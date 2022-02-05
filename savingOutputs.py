import pandas as pd
import os


def save_dicts(filename, dicos, cols, index):
    """
    Function to save a list of dictionaries for scores of different experiments in a csv file
    :param filename: filename for the file that will be created
    :param dicos: list of dictionaries for each
    :param cols: list containing the names of the columns for the csv file
    :param index: list containing the names/numbers of the rows for the csv file
    """
    df = pd.DataFrame(columns=cols)
    for dico in dicos:
        df = df.append(dico, ignore_index=True)

    df.index = index
    df.to_csv(filename)


def save_dicts_perms(filename, dicos, index):
    """
    Function to save a list of dictionaries of *lists of* scores of different experiments in a csv file
    :param filename: filename for the file that will be created
    :param dicos: list of dictionaries for each
    :param index: list containing the names/numbers of the rows for the csv file
    """
    df = pd.DataFrame(dicos)
    df.index = index
    df.to_csv(filename)
