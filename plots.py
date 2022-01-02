from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import os


class Plotter():
    """ This class enables us to save plots of the results obtained in our 
    machine learning process. Its main utility is to ease the handling of folder path
    and avoid code repetition.
    @plot_dir : directory in which all plots will be saved
    @subject_ids : range of the subjects ids (e.g. range(2,5) if we want to observe subjects 2,3 and 4)
    @cv_scores_dir : directory in which to save plots for the cross validation (plots for the cross 
    validation scores will be saved in : self.plot_dir/cv_scores_dir)
    @p_values_dir : directory in which to save plots for the p-values
    @perms_scores_dir : directory in which to save plots for the permutations """

    def __init__(self, plot_dir, subject_ids, cv_scores_dir="cv_scores", p_values_dir="p_values",
                 perms_scores_dir="perms_scores", color = cm.Spectral):
        self.plot_dir = plot_dir
        self.cv_scores_dir = cv_scores_dir
        self.p_values_dir = p_values_dir
        self.perms_scores_dir = perms_scores_dir
        self.subject_ids = subject_ids
        self.color = color
        
        # create the necessary directories
        for di in [cv_scores_dir, p_values_dir, perms_scores_dir]:
            if not os.path.exists(plot_dir + "/" + di):
                os.makedirs(plot_dir + "/" + di)

    def plot_and_save(self, dict_data, label, sub_dir, ylabel, chance_level=False):
        """ Utilitary function that plots y along the self.subject_ids axis with a legend
        and saves it in self.plot_dir/sub_dir/
        @param y: axis to plot
        @param label: legend of the plot
        @param sub_dir: sub directory in which to put the figure """
        plot_df = pd.DataFrame(dict_data, index = ["Right", "Left"])
        plot_df.plot(kind='bar', colormap= self.color)
        if chance_level: plt.axhline(0.25, label="chance level", color="black", alpha=0.5)
        plt.legend()
        plt.title(label)
        plt.xlabel("subject id")
        plt.ylabel(ylabel)
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg")
        plt.close()

    def plot_cv_pval(self, df, ylabel, sub_dir, chance_level=False):
        """ function to plot the results of the cross validation. these are plotted along the subjects axis
        @param filename: the file in which the needed dataframe is located """

        dict_data = {"audition": map(np.mean, [df.iloc[0:]["aud_V5_R"], df.iloc[0:]["aud_V5_L"]]),
                     "vision": map(np.mean, [df.iloc[0:]["vis_V5_R"], df.iloc[0:]["vis_V5_L"]])}
        self.plot_and_save(dict_data, "Decoding in V5", sub_dir, ylabel, chance_level=chance_level)

        dict_data = {"audition": map(np.mean, [df.iloc[0:]["aud_PT_R"], df.iloc[0:]["aud_PT_L"]])}
        self.plot_and_save(dict_data, "Decoding in PT", sub_dir, ylabel, chance_level=chance_level)

        if ylabel == "cv score":
            dict_data = {"audition trained on vision": map(np.mean, [df.iloc[0:]["aud_vis_V5_R"], df.iloc[0:]["aud_vis_V5_L"]]),
                         "vision trained on audition": map(np.mean, [df.iloc[0:]["vis_aud_V5_R"], df.iloc[0:]["vis_aud_V5_L"]])}
            self.plot_and_save(dict_data, "Decoding across modalities in V5", sub_dir, ylabel, chance_level=chance_level)


    def plot_perms_scores(self, df, n_perms):
        """ function to plot the results of the permutations. these are plotted along the subjects axis.
        as we often have a ton of permutations, we plot the mean, var, max and min of the permutations scores
        for each subjects 
        @param filename: the file in which the needed dataframe is located
        @param n_perms: the number of permutations """
        ylabel = "perm score"

        audRV5 = [str_to_array(df.iloc[i]["aud_V5_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audLV5 = [str_to_array(df.iloc[i]["aud_V5_L"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visRV5 = [str_to_array(df.iloc[i]["vis_V5_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visLV5 = [str_to_array(df.iloc[i]["vis_V5_L"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audRPT = [str_to_array(df.iloc[i]["aud_PT_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audLPT = [str_to_array(df.iloc[i]["aud_PT_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]

        dict_data = {"audition": map(np.mean, [audRV5, audLV5]),
                     "vision": map(np.mean, [visRV5, visLV5])}
        self.plot_and_save(dict_data, "Decoding in V5", self.perms_scores_dir, ylabel, chance_level = True)

        dict_data = {"audition": map(np.mean, [audRPT, audLPT])}
        self.plot_and_save(dict_data, "Decoding in PT", self.perms_scores_dir, ylabel, chance_level = True)       

def str_to_array(str_array, length):
    vals = [0] * length
    i = 0
    for v in str_array.split(" "):
        if "\n" in v:
            vals[i] = float(v[0:-2])
            i += 1
        elif v != "":
            vals[i] = float(v)
            i += 1
    return vals
