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
                 perms_scores_dir="perms_scores", bootstrap_dir = "bootstrap", color = cm.Spectral):
        self.plot_dir = plot_dir
        self.subject_ids = subject_ids
        self.cv_scores_dir = cv_scores_dir
        self.p_values_dir = p_values_dir
        self.perms_scores_dir = perms_scores_dir
        self.bootstrap_dir = bootstrap_dir
        self.color = color
        self.translation = {"vis" : ["vision", "visual"], "aud" : ["audition", "auditive"], "R" : "right", "L" : "left"}
        
        # create the necessary directories
        for di in [cv_scores_dir, p_values_dir, perms_scores_dir, bootstrap_dir]:
            if not os.path.exists(plot_dir + "/" + di):
                os.makedirs(plot_dir + "/" + di)


    def save(self, label, sub_dir, ylabel, xlabel = "subject id"):
        plt.legend()
        plt.title(label, wrap = True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg")
        plt.close()


    def bar_plot(self, dict_data, chance_level=False):
        plot_df = pd.DataFrame(dict_data, index = ["Right", "Left"])
        plot_df.plot(kind='bar', colormap= self.color)
        if chance_level: plt.axhline(0.25, label="chance level", color="black", alpha=0.5)
        

    def plot_cv_score(self, df, chance_level=False):
        """ function to plot the results of the cross validation. these are plotted along the subjects axis
        @param filename: the file in which the needed dataframe is located """

        dict_data = {"audition": [df["aud_V5_R"][0], df["aud_V5_L"][0]],
                     "vision": [df["vis_V5_R"][0], df["vis_V5_L"][0]]}
        self.bar_plot(dict_data, chance_level=chance_level)
        self.save("Decoding in V5", self.cv_scores_dir, "cv score")

        dict_data = {"audition": [df["aud_PT_R"][0], df["aud_PT_L"][0]]}
        self.bar_plot(dict_data, chance_level=chance_level)
        self.save("Decoding in PT", self.cv_scores_dir, "cv score")

        dict_data = {"audition trained on vision": [df["aud_vis_V5_R"][0], df["aud_vis_V5_L"][0]],
                     "vision trained on audition": [df["vis_aud_V5_R"][0], df["vis_aud_V5_L"][0]]}
        self.bar_plot(dict_data, chance_level=chance_level)
        self.save("Decoding across modalities in V5", self.cv_scores_dir, "cv score")


    def generate_title(self, label, pval):
        title = "bootstrap for "
        train = ""
        if label [4:7] in self.translation: train = " when training on "+self.translation[label[4:7]][0]
        return title + self.translation[label[:3]][1] + " motion in "+ self.translation[label[-1:]] +" " + label[-4:-2] + train +" (estimated p-value = "+str(round(pval, 4))+")"


    def plot_bootstrap(self, df_bootstrap, df_group_results, pvals, n_bins):
        colors = self.color(np.linspace(0, 1, 10))
        for modality in df_bootstrap:
            color = colors[0] if "aud" == modality[:3] else colors[9]
            plt.hist(df_bootstrap[modality], bins = n_bins, color=color)
            plt.axvline(df_group_results[modality][0], label = "group-level score", color = "green")
            title = self.generate_title(modality, pvals[modality])
            self.save(title, self.bootstrap_dir, "bootstrap number", xlabel="score")             


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
        self.bar_plot(dict_data, chance_level=True)
        self.save("Decoding in V5", self.perms_scores_dir, ylabel)

        dict_data = {"audition": map(np.mean, [audRPT, audLPT])}
        self.bar_plot(dict_data, chance_level=True)
        self.save("Decoding in PT", self.perms_scores_dir, ylabel)       


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
