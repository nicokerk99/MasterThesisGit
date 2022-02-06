from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import seaborn as sns
import os
from utility import *


class Plotter():
    """ This class enables us to save plots of the results obtained in our 
    machine learning process. Its main utility is to ease the handling of folder path
    and avoid code repetition.
    @plot_dir : directory in which all plots will be saved
    @subject_ids : range of the subjects ids (e.g. range(2,5) if we want to observe subjects 2,3 and 4)
    @cv_scores_dir : directory in which to save plots for the cross validation (plots for the cross 
    validation scores will be saved in : self.plot_dir/cv_scores_dir)
    @p_values_dir : directory in which to save plots for the p-values
    @perms_scores_dir : directory in which to save plots for the permutations
    @bootstrap_dir : directory in which to save plots regarding bootstrap
    @color : color for the different plots (must be one of matplotlib.cm's colormaps) """

    def __init__(self, plot_dir, subject_ids, cv_scores_dir="cv_scores", p_values_dir="p_values",
                 perms_scores_dir="perms_scores", bootstrap_dir="bootstrap"):
        self.plot_dir = plot_dir
        self.subject_ids = subject_ids
        self.cv_scores_dir = cv_scores_dir
        self.p_values_dir = p_values_dir
        self.perms_scores_dir = perms_scores_dir
        self.bootstrap_dir = bootstrap_dir
        self.color = ListedColormap(cm.get_cmap("brg")(np.linspace(0, 0.5, 256)))
        colors = self.color(np.linspace(0, 1, 3))
        self.modality_to_color = {"vis": colors[2], "aud": colors[0], "cro": colors[1]}
        self.name_to_color = {"Vision": colors[2], "Audition": colors[0], "Cross-modal": colors[1]}
        self.translation = {"vis": ["vision", "visual"], "aud": ["audition", "auditive"], "cro": "cross-modal",
                            "R": "right", "L": "left"}

        # create the necessary directories
        for di in [cv_scores_dir, p_values_dir, perms_scores_dir, bootstrap_dir]:
            if not os.path.exists(plot_dir + "/" + di):
                os.makedirs(plot_dir + "/" + di)

        plt.rcParams.update({'font.size': 15})

    def save(self, label, sub_dir, ylabel, xlabel="subject id", legend=True):
        """ function that adds the legend, title, label axes and saves a plot in self.plot_dir/sub_dir/label.jpg.
        @param label : the title of the plot and name of the file in which the plot will be saved
        @param sub_dir : the directory in which the plot will be saved
        @param ylabel : label for the y axis
        @param xlabel : label for the x axis
        @param legend : boolean to tell if need of a legend or not """

        if legend :
            plt.legend(loc="lower center")
        plt.title(label, wrap=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg", bbox_inches='tight')
        plt.close()

    def bar_plot_within_modal(self, dict_data, chance_level):
        """ plots a dictionnary as a bar plot with index ["Right", "Left"]
        @param dict_data : the dictionnary containing the data to be plotted
        @param chance_level : set to True if you want a line y = 0.25 to be added to the plot """

        plot_df = pd.DataFrame(dict_data, index=["Left", "Right"])
        plot_df.plot(kind='bar', colormap=self.color)
        if chance_level:
            plt.axhline(0.25, label="chance level", color="black", alpha=0.5)
        plt.xticks(rotation=0)

    def bar_plot_cross_modal(self, data_list, chance_level):
        """ plots a dictionnary as a bar plot with index ["Right", "Left"]
        @param data_list : list containing the data to be plotted
        @param chance_level : set to True if you want a line y = 0.25 to be added to the plot """

        plt.bar(["Left", "Right"], data_list, color=self.modality_to_color["cro"], width=0.3)
        if chance_level:
            plt.axhline(0.25, label="chance level", color="black", alpha=0.5)
        plt.xticks(rotation=0)

    def plot_cv_score(self, df, chance_level=False):
        """ function to plot the results of the cross validation.
        @param df : the dataframe containing the cross val results
        @param chance_level : defaults to False, set to True if you want a line y = 0.25 to be added to the plot """

        ylabel = "CV score"

        dict_data = {"audition": [df["aud_V5_L"][0], df["aud_V5_R"][0]],
                     "vision": [df["vis_V5_L"][0], df["vis_V5_R"][0]]}
        self.bar_plot_within_modal(dict_data, chance_level)
        self.save("Decoding within modality in V5", self.cv_scores_dir, ylabel, xlabel="hemisphere")

        dict_data = {"audition": [df["aud_PT_L"][0], df["aud_PT_R"][0]],
                     "vision": [df["vis_PT_L"][0], df["vis_PT_R"][0]]}
        self.bar_plot_within_modal(dict_data, chance_level)
        self.save("Decoding within modality in PT", self.cv_scores_dir, ylabel, xlabel="hemisphere")

        data = [float(df["cross_V5_L"]),
                float(df["cross_V5_R"])]
        self.bar_plot_cross_modal(data, chance_level)
        self.save("Decoding across modalities in V5", self.cv_scores_dir, ylabel, xlabel="hemisphere")

        data = [float(df["cross_PT_L"]),
                float(df["cross_PT_R"])]
        self.bar_plot_cross_modal(data, chance_level)
        self.save("Decoding across modalities in PT", self.cv_scores_dir, ylabel, xlabel="hemisphere")

    def bar_plot_with_points(self, df, chance_level):
        if df["Region"].nunique() <= 2 :
            plt.figure(figsize=(10, 10))
        else :
            plt.figure(figsize=(23, 10))

        # Draw the bar chart
        ax = sns.catplot(
            data=df,
            kind="bar",
            ci=None,
            x="Region",
            y="Score",
            hue="Modality",
            palette=self.name_to_color,
            alpha=.5,
        )
        g = sns.stripplot(
            data=df,
            x="Region",
            y="Score",
            hue="Modality",
            dodge=True,
            palette=self.name_to_color,
            alpha=0.6,
            size=7
        )
        ax = sns.barplot(
            data=df,
            ci="sd",
            capsize=0.1,
            errcolor="darkslategrey",
            errwidth=2.5,
            x="Region",
            y="Score",
            hue="Modality",
            palette=self.name_to_color,
            alpha=.5,
        )
        g.legend_.remove()

        if chance_level:
            plt.axhline(0.25, label="chance level", color="black", alpha=0.5)

    def plot_cv_score_with_points(self, df, chance_level=False):
        """ function to plot the results of the cross validation, with individual points.
        @param df : the dataframe containing the cross val results
        @param chance_level : defaults to False, set to True if you want a line y = 0.25 to be added to the plot """
        ylabel = "CV score"

        df_within_V5 = verbose_dataframe(df[["aud_V5_L", "aud_V5_R", "vis_V5_L", "vis_V5_R"]])
        self.bar_plot_with_points(df_within_V5, chance_level)
        self.save("Decoding within modality in V5", self.cv_scores_dir, ylabel, xlabel="analysis", legend=False)

        df_within_PT = verbose_dataframe(df[["aud_PT_L", "aud_PT_R", "vis_PT_L", "vis_PT_R"]])
        self.bar_plot_with_points(df_within_PT, chance_level)
        self.save("Decoding within modality in PT", self.cv_scores_dir, ylabel, xlabel="analysis", legend=False)

        df_within_all = verbose_dataframe(df[["aud_V5_L", "aud_V5_R", "vis_V5_L", "vis_V5_R",
                                              "aud_PT_L", "aud_PT_R", "vis_PT_L", "vis_PT_R"]])
        self.bar_plot_with_points(df_within_all, chance_level)
        self.save("Decoding within modality", self.cv_scores_dir, ylabel, xlabel="analysis", legend=False)

        df_cross_V5 = verbose_dataframe(df[["cross_V5_L", "cross_V5_R"]])
        self.bar_plot_with_points(df_cross_V5, chance_level)
        self.save("Decoding across modalities in V5", self.cv_scores_dir, ylabel, xlabel="analysis", legend=False)

        df_cross_PT = verbose_dataframe(df[["cross_PT_L", "cross_PT_R"]])
        self.bar_plot_with_points(df_cross_PT, chance_level)
        self.save("Decoding across modalities in PT", self.cv_scores_dir, ylabel, xlabel="analysis", legend=False)

    def generate_title(self, modality, pval):
        """ function that generates the title for bootstrap plots
        @param modality :  the modality (e.g. "aud_vis_V5_R")
        @param pval : the estimated p-value """

        if modality[:3] == "cro":
            beginning = "Bootstrap for cross-modal decoding in "
        else:
            beginning = "Bootstrap for " + self.translation[modality[:3]][1] + " motion in "
        return beginning + self.translation[modality[-1:]] + " " + modality[-4:-2] \
               + " (estimated p-value = " + str(round(pval, 6)) + ")"

    def plot_bootstrap(self, df_bootstrap, df_group_results, pvals, n_bins):
        """ function to plot the bootstrap results. we plot a histogram of the bootstrap results for each modality and 
        add a vertical line that represents the group result for this modality.
        @param df_bootstrap : the dataframe containing the bootstrap scores
        @param df_group_results : the dataframe containing the group results
        @param pvals : a dictionnary containing the estimated p-value for each modality
        @param n_bins : the numbers of bins for the bootstrap histogram"""

        for modality in df_bootstrap:
            color = self.modality_to_color[modality[:3]]
            plt.hist(df_bootstrap[modality], bins=n_bins, color=color)
            plt.axvline(df_group_results[modality][0], label="group-level score", color="green")
            self.save(self.generate_title(modality, pvals[modality]), self.bootstrap_dir, "density", xlabel="score")

    def plot_perms_scores(self, df, n_perms, chance_level=False):
        """ function to plot the results of the permutations. these are plotted along the subjects axis.
        as we often have a ton of permutations, we plot the mean of the permutations scores for each modality 
        @param df : the dataframe containing the permutations scores
        @param n_perms : the number of permutations
        @param chance_level : defaults to False, set to True if you want a line y = 0.25 to be added to the plot """

        ylabel = "perm score"

        audRV5 = [str_to_array(df.iloc[i]["aud_V5_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audLV5 = [str_to_array(df.iloc[i]["aud_V5_L"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visRV5 = [str_to_array(df.iloc[i]["vis_V5_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visLV5 = [str_to_array(df.iloc[i]["vis_V5_L"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audRPT = [str_to_array(df.iloc[i]["aud_PT_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        audLPT = [str_to_array(df.iloc[i]["aud_PT_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visRPT = [str_to_array(df.iloc[i]["vis_PT_R"][1:-1], n_perms) for i in range(len(self.subject_ids))]
        visLPT = [str_to_array(df.iloc[i]["vis_PT_L"][1:-1], n_perms) for i in range(len(self.subject_ids))]

        dict_data = {"audition": map(np.mean, [audLV5, audRV5]),
                     "vision": map(np.mean, [visLV5, visRV5])}
        self.bar_plot_within_modal(dict_data, chance_level)
        self.save("Decoding in V5", self.perms_scores_dir, ylabel)

        dict_data = {"audition": map(np.mean, [audLPT, audRPT]),
                     "vision": map(np.mean, [visLPT, visRPT])}
        self.bar_plot_within_modal(dict_data, chance_level)
        self.save("Decoding in PT", self.perms_scores_dir, ylabel)

    def plot_confusion_matrix(self):
        pass


def plot_average_voxel_intensities(maps, classes, n_subjects):
    region = "V5_R"
    colors = ["red", "tomato", "coral", "orange", "deepskyblue", "cyan", "blue", "royalblue"]
    n_voxels = maps[0]["vis"][0][region].shape[1]
    mean_aud = dict()
    mean_vis = dict()

    for cla in classes:
        mean_aud[cla] = np.zeros(n_voxels)
        mean_vis[cla] = np.zeros(n_voxels)

    for i in range(n_subjects):
        for j, cla in enumerate(classes):
            mean_vis[cla] += np.mean(maps[i]["vis"][0][region][j * 12:(j + 1) * 12], axis=0) / n_subjects
            mean_aud[cla] += np.mean(maps[i]["aud"][0][region][j * 12:(j + 1) * 12], axis=0) / n_subjects

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(12, 8))

    idx = 0
    for cla in classes:
        plt.plot(range(n_voxels), mean_vis[cla], label="vis - " + cla, color=colors[idx])
        idx += 1

    for cla in classes:
        plt.plot(range(n_voxels), mean_aud[cla], label="aud - " + cla, color=colors[idx])
        idx += 1

    plt.xlabel("voxel id")
    plt.ylabel("intensity")
    plt.title("Average voxel intensities for " + region)
    plt.legend()
    plt.savefig("plots/average_voxel_intensities" + region + ".png")


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
