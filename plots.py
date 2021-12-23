from matplotlib import pyplot as plt
from matplotlib import cm
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
        self.subject_ids = range(len(subject_ids))
        self.colormaps = [cm.Spectral, cm.Set1, cm.Set2, cm.tab10, cm.Set3]
        self.translation = {"vis" : ["vision", "visual"], "aud" : ["audition", "auditive"], "R" : "right", "L" : "left"}

        # create the necessary directories
        for di in [cv_scores_dir, p_values_dir, perms_scores_dir]:
            if not os.path.exists(plot_dir + "/" + di):
                os.makedirs(plot_dir + "/" + di)


    def save_plot(self, label, sub_dir, ylabel, chance_level = False):
        """ Utilitary function that plots y along the self.subject_ids axis with a legend
        and saves it in self.plot_dir/sub_dir/
        @param y: axis to plot
        @param label: legend of the plot
        @param sub_dir: sub directory in which to put the figure """
        if chance_level: plt.plot(self.subject_ids, [0.25]*len(self.subject_ids), label = "chance level", color = "black", alpha = 0.5)
        plt.legend()
        plt.title(self.generate_title(label, ylabel))
        plt.xlabel("subject id")
        plt.ylabel(ylabel)
        plt.savefig(self.plot_dir + "/" + sub_dir + "/" + label + ".jpg")
        plt.close()


    def generate_title(self, label, plot_type):
        title = ""
        train = ""
        if plot_type == "perm score": 
            title = label.split('_')[0]+" "
            label = label[len(title):]

        if label [4:7] in self.translation: 
            train = " when training on "+self.translation[label[4:7]][0]
        
        title = title + "Decoding " + self.translation[label[:3]][1] + " motion direction in "+ label[-2:] + train
        
        if plot_type == "p-values": return "P-value when "+title 
        else : return title


    def plot_cv_pval(self, filename, ylabel, sub_dir, chance_level = False):
        """ function to plot the results of the cross validation. these are plotted along the subjects axis
        @param filename: the file in which the needed dataframe is located """
        df = pd.read_csv(filename)
        keys = df.keys()[1:]
        colors = self.colormaps[:len(keys)]

        i = 0
        while i < len(keys):
            keyR = keys[i]
            keyL = keys[i+1]
            plot_data = pd.DataFrame({"R" : df.iloc[0:][keyR], "L" : df.iloc[0:][keyL]}, index = self.subject_ids)
            plot_data.plot(kind = 'bar', colormap = colors[int(i/2)])
            self.save_plot(keyR[:-2], sub_dir, ylabel, chance_level = chance_level)
            i+=2


    def plot_perms_scores(self, filename, n_perms):
        """ function to plot the results of the permutations. these are plotted along the subjects axis.
        as we often have a ton of permutations, we plot the mean, var, max and min of the permutations scores
        for each subjects 
        @param filename: the file in which the needed dataframe is located
        @param n_perms: the number of permutations """
        df = pd.read_csv(filename)
        ylabel = "perm score"
        keys = df.keys()[1:]
        colors = self.colormaps[:len(keys)]

        i = 0
        while i < len(keys):
            keyR = keys[i]
            keyL = keys[i+1]
            
            valR = [str_to_array(df.iloc[i][keyR][1:-1], n_perms) for i in range(len(self.subject_ids))]
            valL = [str_to_array(df.iloc[i][keyL][1:-1], n_perms) for i in range(len(self.subject_ids))]

            plot_data = pd.DataFrame({"R" : list(map(stats.mean, valR)), "L" : list(map(stats.mean, valL))}, index = self.subject_ids)
            plot_data.plot(kind = 'bar', colormap = colors[int(i/2)])
            self.save_plot("mean_" + keyR[:-2], self.perms_scores_dir, ylabel)
            
            plot_data = pd.DataFrame({"R" : list(map(stats.variance, valR)), "L" : list(map(stats.variance, valL))}, index = self.subject_ids)
            plot_data.plot(kind = 'bar', colormap = colors[int(i/2)])
            self.save_plot("var_" + keyR[:-2], self.perms_scores_dir, ylabel)
            
            plot_data = pd.DataFrame({"R" : list(map(min, valR)), "L" : list(map(min, valL))}, index = self.subject_ids)
            plot_data.plot(kind = 'bar', colormap = colors[int(i/2)])
            self.save_plot("min_" + keyR[:-2], self.perms_scores_dir, ylabel)

            plot_data = pd.DataFrame({"R" : list(map(max, valR)), "L" : list(map(max, valL))}, index = self.subject_ids)
            plot_data.plot(kind = 'bar', colormap = colors[int(i/2)])
            self.save_plot("max_" + keyR[:-2], self.perms_scores_dir, ylabel)
            i+=2



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
