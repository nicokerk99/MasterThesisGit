from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import pylab
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import seaborn as sns
from utility import *
from load_data import *
import math


class Plotter:
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

    def __init__(self, plot_dir, subject_ids, cv_scores_dir="cv_scores", conf_matrixes_dir="conf_matrixes",
                 bootstrap_dir="bootstrap"):
        self.plot_dir = plot_dir
        self.subject_ids = subject_ids
        self.cv_scores_dir = cv_scores_dir
        self.bootstrap_dir = bootstrap_dir
        self.conf_matrixes_dir = conf_matrixes_dir
        self.color = ListedColormap(cm.get_cmap("brg")(np.linspace(0, 0.5, 256)))
        colors = self.color(np.linspace(0, 1, 3))
        self.modality_to_color = {"vis": colors[2], "aud": colors[0], "cro": colors[1]}
        self.analysis_to_color = {"Vision": colors[2], "Audition": colors[0], "Cross-modal": colors[1]}
        self.translation = {"vis": ["vision", "visual"], "aud": ["audition", "auditive"], "cro": "cross-modal",
                            "R": "right", "L": "left"}

        # create the necessary directories
        for di in [cv_scores_dir, bootstrap_dir, conf_matrixes_dir]:
            create_directory(plot_dir + "/" + di)

        plt.rcParams.update({'font.size': 12})

    def save(self, label, sub_dir, ylabel, xlabel="subject id", legend=True):
        """ function that adds the legend, title, label axes and saves a plot in self.plot_dir/sub_dir/label.jpg.
        @param label : the title of the plot and name of the file in which the plot will be saved
        @param sub_dir : the directory in which the plot will be saved
        @param ylabel : label for the y axis
        @param xlabel : label for the x axis
        @param legend : boolean to tell if need of a legend or not """

        if legend is not None:
            if legend is True:
                plt.legend(loc="lower center")
            else:
                plt.legend(loc=legend)
        #plt.title(label, wrap=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        label = label.replace(" ", "_")
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg", bbox_inches='tight')
        plt.close()

    def bar_plot_with_points(self, df, chance_level, pvals=None, compare=False, hue="Modality",is_variance=False):
        if not compare :
            if df["Region"].nunique() <= 2:
                plt.figure(figsize=(10, 10))
            else:
                plt.figure(figsize=(23, 10))

            x = "Region"
            hue_order = ["Vision", "Audition"]
            palette = self.analysis_to_color
            if df[hue].nunique() <= 1:
                hue_order = ["Cross-modal"]

        else:
            plt.figure(figsize=(40, 10))
            x = "Analysis"
            hue_order = compare
            palette = None

        # Draw the bar chart
        sns.catplot(
            data=df,
            kind="bar",
            ci=None,
            x=x,
            y="Score",
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            alpha=.7,
        )
        if not compare :
            g = sns.stripplot(
                data=df,
                x=x,
                y="Score",
                hue=hue,
                hue_order=hue_order,
                dodge=True,
                palette=palette,
                alpha=0.6,
                size=7
            )
        bplot = sns.barplot(
            data=df,
            ci="sd",
            capsize=0.1,
            errcolor="black",
            errwidth=1.0,
            x=x,
            y="Score_mean_dev",
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            alpha=0.1,
        )
        bplot.legend_.remove()

        if pvals is not None:
            i = 0
            for bar in bplot.patches[:len(pvals)]:
                star = stars(pvals[i])
                if star != "ns":
                    bplot.annotate(star,
                                   (bar.get_x() + bar.get_width() / 2, 0.2),
                                   ha="center", va="center",
                                   size=12, xytext=(0, 8),
                                   textcoords="offset points",
                                   color="white")
                i += 1
            if len(pvals) > 8 :
                plt.xticks(rotation=90)
        else :
            plt.xticks(rotation=90)

        if is_variance:
            plt.ylim(0, 0.12)
        else:
            plt.ylim(0.2, 0.5)

        if chance_level:
            plt.axhline(0.25, label="chance level", color="black", alpha=0.5)

    def plot_cv_score_with_points(self, df, pvals, chance_level=False):
        """ function to plot the results of the cross validation, with individual points.
        @param df : the dataframe containing the cross val results
        @param chance_level : defaults to False, set to True if you want a line y = 0.25 to be added to the plot """
        ylabel = "CV score"

        for region in ["PT", "V5"]:
            labels = ["aud_" + region + "_L", "aud_" + region + "_R", "vis_" + region + "_L", "vis_" + region + "_R"]
            labels_pvals = ["vis_" + region + "_L", "vis_" + region + "_R", "aud_" + region + "_L",
                            "aud_" + region + "_R"]
            df_within = verbose_dataframe(df[labels], self.subject_ids)
            self.bar_plot_with_points(df_within, chance_level, pvals=[pvals[l] for l in labels_pvals])
            self.save("Decoding within modality in " + region, self.cv_scores_dir, ylabel, xlabel="analysis",
                      legend=None)

            labels = ["cross_" + region + "_L", "cross_" + region + "_R"]
            df_cross = verbose_dataframe(df[labels], self.subject_ids)
            self.bar_plot_with_points(df_cross, chance_level, pvals=[pvals[l] for l in labels])
            plt.ylim(0.2, 0.5)
            self.save("Decoding across modalities in " + region, self.cv_scores_dir, ylabel, xlabel="analysis",
                      legend=None)

        labels = ["aud_V5_L", "aud_V5_R", "vis_V5_L", "vis_V5_R", "aud_PT_L", "aud_PT_R", "vis_PT_L", "vis_PT_R"]
        labels_pvals = ["vis_V5_L", "vis_V5_R", "vis_PT_L", "vis_PT_R", "aud_V5_L", "aud_V5_R", "aud_PT_L", "aud_PT_R"]
        df_within_all = verbose_dataframe(df[labels], self.subject_ids)
        self.bar_plot_with_points(df_within_all, chance_level, pvals=[pvals[k] for k in labels_pvals])
        self.save("Decoding within modality", self.cv_scores_dir, ylabel, xlabel="analysis", legend=None)

    def generate_title(self, begin, modality, pval):
        """ function that generates the title for bootstrap plots
        @param modality :  the modality (e.g. "aud_V5_R")
        @param pval : the estimated p-value """

        if modality[:3] == "cro":
            beginning = begin + " for cross-modal decoding in "
        else:
            beginning = begin + " for " + self.translation[modality[:3]][1] + " motion in "
        title = beginning + self.translation[modality[-1:]] + " " + modality[-4:-2]

        if pval > 0:
            return title + " (estimated p-value = " + str(min(round(pval, 6), 1)) + ")"
        else:
            return title

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
            title = self.generate_title("Bootstrap", modality, pvals[modality])
            self.save(title, self.bootstrap_dir, "density", xlabel="score")

    def plot_group_confusion_matrix(self, group_cfm, classes):
        for modality in group_cfm:
            mean_cfm = np.zeros((len(classes), len(classes)))
            var_cfm = np.zeros((len(classes), len(classes)))
            cfm = group_cfm[modality]
            for i in range(len(classes)):
                for j in range(len(classes)):
                    mean_cfm[i][j] = np.mean(cfm[i][j])
                    var_cfm[i][j] = np.var(cfm[i][j])

            for stat, values in zip(["mean", "variance"], [mean_cfm, var_cfm]):
                df = pd.DataFrame(values, index=classes, columns=classes)
                pylab.figure(figsize=(8, 8))
                sns.heatmap(df, linewidth=1, annot=True, cmap=cm.YlOrRd)
                title = self.generate_title("Confusion Matrix " + stat, modality, -1)
                self.save(title, self.conf_matrixes_dir, "true label", "predicted label", None)

    def get_score_per_analysis(self, val_sc_df, group_by, group_by_values, masks_exist):
        score_per_analysis = dict()
        str_group_by_values = [str(val) for val in group_by_values]

        for analysis in val_sc_df:
            score_per_analysis[analysis] = dict()
            for i, subj in enumerate(self.subject_ids):
                if masks_exist[analysis.split("_", 1)[1]][subj]:
                    combs = val_sc_df[analysis][subj].replace(" ", "").replace("{", "").replace("}", "").replace("\'",
                                                                                                                 "").split(
                        ",")
                    for comb in combs:
                        index = 0
                        new_key = ""
                        tmp = comb.split(":")
                        score = float(tmp[1])
                        indiv_params = tmp[0].split("-")
                        if indiv_params[-1] == "cg":
                            indiv_params[-2] += "cg"
                            indiv_params = indiv_params[:-1]
                        for param in indiv_params:
                            key = param.split("__")[1]
                            tmp_ = param.split("=")
                            name = tmp_[0].split("__")[1]
                            value = tmp_[1]
                            if name != group_by:
                                new_key += " - " + key
                            else:
                                index = str_group_by_values.index(value)

                        if len(new_key) > 0:
                            new_key = new_key.split(" - ", 1)[1]

                        if new_key in score_per_analysis[analysis]:
                            score_per_analysis[analysis][new_key][i, index] = score
                        else:
                            score_per_analysis[analysis][new_key] = np.zeros(
                                (len(self.subject_ids), len(group_by_values)))
                            score_per_analysis[analysis][new_key][i, 0] = score

        for analysis in score_per_analysis:
            for key in score_per_analysis[analysis]:
                tab = score_per_analysis[analysis][key]
                tab = tab.astype('float')
                tab[tab == 0] = 'nan'
                score_per_analysis[analysis][key] = np.nanmean(tab, axis=0)

        return score_per_analysis

    def plot_validation_scores_hyper_param(self, val_sc_df, x_label, x_values, masks_exist, chance_level=True,
                                           log10_scale=False):
        """
        function to plot average validation scores accumulated by GridSearch along the process
        :param val_sc_df: dataframe as given by load_data.retrieve_val_scores
        :param x_label: string with the paramater to put in x-axis
        :param x_values: values that the parameter will take
        """
        val_dir = self.plot_dir + "/validation_scores"
        create_directory(val_dir)
        score_per_analysis = self.get_score_per_analysis(val_sc_df, x_label, x_values, masks_exist)
        for modality in score_per_analysis:
            plt.figure(figsize=(12, 8))
            for params in score_per_analysis[modality]:
                if log10_scale:
                    plt.plot([math.log10(x) for x in x_values], score_per_analysis[modality][params], label=params)
                else:
                    plt.plot(x_values, score_per_analysis[modality][params], label=params)
            plt.ylim(0.2, 0.5)
            if chance_level:
                plt.axhline(0.25, color="black", alpha=0.5)
            title = self.generate_title("Validation score", modality, pval=-1)
            x_lab = "log10(" + x_label + ")" if log10_scale else x_label
            self.save(title, "validation_scores", "validation score", x_lab, legend="best")

    def plot_tests_scores_from_different_folders(self, folder_names, labels, title, hue, p_vals=False):
        """
        plot test scores as bar plots for different classifiers
        :param folder_names: the output folders to retrieve the scores
        :param labels: the names of the classifiers to put on the legend of the plot
        :return:
        """
        labels_within = ["aud_V5_L", "aud_V5_R", "vis_V5_L", "vis_V5_R", "aud_PT_L", "aud_PT_R"]
        labels_cross = ["cross_V5_L", "cross_V5_R", "cross_PT_L", "cross_PT_R"]
        big_within_df = pd.DataFrame()
        big_cross_df = pd.DataFrame()
        p_vals_within = [None] * len(folder_names)
        for i, name in enumerate(folder_names):
            cv_df = retrieve_cv_metric(name, "accuracy")
            pvals = retrieve_pvals(name, default_keys=cv_df.columns)
            labels_pvals = ["vis_V5_L", "vis_V5_R", "aud_V5_L", "aud_V5_R", "aud_PT_L", "aud_PT_R"]
            p_vals_within[i] = ([pvals[lab] for lab in labels_pvals]) if p_vals else None

            df_within = verbose_dataframe(cv_df[labels_within], self.subject_ids, compare=True)
            df_within[hue] = [labels[i]]*df_within.shape[0]
            big_within_df = pd.concat([big_within_df, df_within])

            df_cross = verbose_dataframe(cv_df[labels_cross], self.subject_ids, compare=True)
            df_cross[hue] = [labels[i]]*df_cross.shape[0]
            big_cross_df = pd.concat([big_cross_df, df_cross])

        p_vals_ordered = interleave_lists(p_vals_within) if p_vals else None

        self.bar_plot_with_points(big_within_df, True, pvals=p_vals_ordered, compare=labels, hue=hue)
        self.save(title+" within modality", "", "Accuracy", xlabel="Analysis", legend=None)

        self.bar_plot_with_points(big_cross_df, True, pvals=None, compare=labels, hue=hue)
        self.save(title+" across modalities", "", "Accuracy", xlabel="Analysis", legend=None)

    def plot_accuracy_std_from_different_folders(self, folder_names, labels, title, hue):
        labels_ = {}
        labels_["within"] = ["aud_V5_L", "aud_V5_R", "vis_V5_L", "vis_V5_R", "aud_PT_L", "aud_PT_R"]
        labels_["cross"] = ["cross_V5_L", "cross_V5_R", "cross_PT_L", "cross_PT_R"]
        for mode in ["within", "cross"]:
            big_df = pd.DataFrame()
            for i, name in enumerate(folder_names):
                df = pd.read_csv(name+"var_"+mode+".csv", index_col=0)
                df = df.applymap(lambda x : np.sqrt(x))
                # new_cols = df.index[1:]
                new_cols = labels_[mode]
                temp_df = pd.DataFrame(columns=new_cols, index=[1], dtype=float)
                temp_df[new_cols] = df[df.columns[0]][labels_[mode]].values

                df_mode = verbose_dataframe(temp_df[new_cols], temp_df.index, compare=True)
                df_mode[hue] = [labels[i]]*df_mode.shape[0]

                big_df = pd.concat([big_df, df_mode], ignore_index=True)
            
            self.bar_plot_with_points(big_df, False, compare=labels, hue=hue, is_variance=True)
            self.save("Accuracy standard deviation for "+title.lower()+" "+mode+" modality", "", "standard deviation", xlabel="Analysis", legend=None)


# function that interleaves element from multiple lists
def interleave_lists(lists):
    final_list = []
    for i in range(len(lists[0])):
        for j in range(len(lists)):
            final_list.append(lists[j][i])
    return final_list


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


def plot_normalize_voxel_intensities(maps, classes):
    region = "V5_R"
    colors = ["deepskyblue", "cyan", "blue", "royalblue"]
    n_voxels = maps[0]["vis"][0][region].shape[1]
    mean_aud = dict()

    for cla in classes:
        mean_aud[cla] = np.zeros(n_voxels)

    for j, cla in enumerate(classes):
        for run in range(11) :
            mean_aud[cla] += maps[1]["aud"][0][region][(run * 4)+j]
        mean_aud[cla] = np.divide(mean_aud[cla], 11)

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(12, 8))

    idx = 0

    for cla in classes:
        plt.plot(range(n_voxels), mean_aud[cla], label="aud - " + cla, color=colors[idx])
        idx += 1

    plt.xlabel("voxel id")
    plt.ylabel("intensity")
    plt.title("Original voxel intensities for audition in " + region)
    plt.legend()
    plt.savefig("plots/normalize_voxel_intensities" + region + ".png")


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


def stars(pval):
    if pval > 0.05: return "ns"
    if pval <= 0.0001: return "****"
    if pval <= 0.001: return "***"
    if pval <= 0.01:
        return "**"
    else:
        return "*"
