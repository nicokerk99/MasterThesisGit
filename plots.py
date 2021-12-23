from matplotlib import pyplot as plt
import statistics as stats
import pandas as pd
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
                 perms_scores_dir="perms_scores"):
        self.plot_dir = plot_dir
        self.cv_scores_dir = cv_scores_dir
        self.p_values_dir = p_values_dir
        self.perms_scores_dir = perms_scores_dir
        self.subject_ids = subject_ids
        self.colors = ["red", "red", "blue", "blue", "green", "green", "yellow", "yellow", "brown", "brown"]
        self.translation = {"vis" : "vision", "aud" : "audition", "R" : "right", "L" : "left"}

        # create the necessary directories
        for di in [cv_scores_dir, p_values_dir, perms_scores_dir]:
            if not os.path.exists(plot_dir + "/" + di):
                os.makedirs(plot_dir + "/" + di)

    def plot_and_save(self, y, label, sub_dir, chance_level = False, color = "orange"):
        """ Utilitary function that plots y along the self.subject_ids axis with a legend
        and saves it in self.plot_dir/sub_dir/
        @param y: axis to plot
        @param label: legend of the plot
        @param sub_dir: sub directory in which to put the figure """
        plt.bar(self.subject_ids, y, label=label.replace("_", " "), color = color)
        if chance_level: plt.plot(self.subject_ids, [0.25]*len(self.subject_ids), label = "chance level", color = "black")
        plt.legend()
        if label[:3] in self.translation:
            train = ""
            if label [4:7] in self.translation: train = " when training on "+self.translation[label[4:7]] 
            plt.title("Decoding of " + self.translation[label[:3]] + " in "+ label[-4:-2] + train + " (" +self.translation[label[-1]] + " hemisphere)")
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg")
        plt.close()

    def plot_cv_scores(self, filename):
        """ function to plot the results of the cross validation. these are plotted along the subjects axis
        @param filename: the file in which the needed dataframe is located """
        df = pd.read_csv(filename)

        for k, c in zip(df.keys()[1:], self.colors[:len(df.keys()[1:])]):
            self.plot_and_save(df.iloc[0:][k], k, self.cv_scores_dir, chance_level = True, color = c)

    def plot_p_values(self, filename):
        """ function to plot the p-values. these are plotted along the subjects axis
        @param filename: the file in which the needed dataframe is located """
        df = pd.read_csv(filename)

        for k, c in zip(df.keys()[1:], self.colors[:len(df.keys()[1:])]):
            self.plot_and_save(df.iloc[0:][k], k, self.p_values_dir, color = c)

    def plot_perms_scores(self, filename, n_perms):
        """ function to plot the results of the permutations. these are plotted along the subjects axis.
        as we often have a ton of permutations, we plot the mean, var, max and min of the permutations scores
        for each subjects 
        @param filename: the file in which the needed dataframe is located
        @param n_perms: the number of permutations """
        df = pd.read_csv(filename)

        for k in df.keys()[1:]:
            values = [str_to_array(df.iloc[i][k][1:-1], n_perms) for i in range(len(self.subject_ids))]

            self.plot_and_save(list(map(stats.mean, values)), "mean_" + k, self.perms_scores_dir)
            self.plot_and_save(list(map(stats.variance, values)), "var_" + k, self.perms_scores_dir)
            self.plot_and_save(list(map(max, values)), "max_" + k, self.perms_scores_dir)
            self.plot_and_save(list(map(min, values)), "min_" + k, self.perms_scores_dir)


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
