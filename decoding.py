import random
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from utility import average_dicos


def permute_labels(labels, gap):
    labels_copy = labels.copy()

    # shuffling but keeping a structure
    for i in range(gap):
        indexes = range(i, len(labels), gap)
        subset_copy = [labels_copy[idx] for idx in indexes]
        random.shuffle(subset_copy)
        labels_copy[i:len(labels):gap] = subset_copy

    return labels_copy


class Decoder:
    """ This class eases the use of machine learning models, cross-validation
    and random permutations significance assessment for neuro-imaging data
    @model : machine learning model that will be used
    @n_classes : the number of classes that will be potentially be predicted
    @n_splits : number of folds for cross validation
    @seed : random seed to make sure each time we run the code, we obtain the same results
    @n_perm : number of permutations to make when inspecting significance """

    def __init__(self, model, n_classes, n_splits, seed, n_perm):
        self.model = model
        self.n_classes = n_classes
        self.n_splits = n_splits
        self.seed = seed
        self.n_perm = n_perm
        self.predictions = dict()

    def return_and_reset_predictions(self, keys):
        """
        :param keys: array of strings to set as keys
        :return: a dictionary containing the amount of predictions done for each class
        """
        to_return = self.predictions.copy()
        self.predictions = dict()
        for key in keys :
            self.predictions[key] = 0
        return to_return

    def update_predictions(self, predictions):
        for pred in predictions :
            self.predictions[pred] += 1

    def cross_validate(self, brain_map, labels, return_model=False):
        """ Attention, this function is based on labels with consecutive, balanced categories, like['U','U','D','D',
        'R','R','L','L']
        :param brain_map: list of maps (size n_samples), which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param return_model: boolean to say if we have to return the fitted model
        :return: the cross-validation score obtained on the data """

        acc = 0
        for ind in range(self.n_splits):
            test_index = range(ind, len(brain_map), self.n_splits)
            train_index = range(0, len(brain_map))
            train_index = [ind for ind in train_index if ind not in test_index]

            X_train, X_test = brain_map[train_index], brain_map[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            self.update_predictions(predictions)
            score = self.model.score(X_test, y_test)
            acc += score

        final_score = acc / self.n_splits

        if return_model:
            self.model.fit(brain_map, labels)  # re-fitting the model on all data
            return final_score, self.model
        else:
            return final_score

    def p_value_random_permutations(self, brain_map, labels, base_score):
        """ Attention, this function is based on labels with consecutive, balanced categories, like['U','U','D','D',
        'R','R','L','L']
        :param brain_map: list of maps (size n_samples), which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param base_score: score obtained with true labels
        :return: the estimated p-value for classification on this data,
         and permutations scores """

        random.seed(self.seed)
        count = 0
        gap = int(len(labels) / self.n_classes)
        scores_perms = np.zeros(self.n_perm)
        for j in range(self.n_perm):
            labels_perm = permute_labels(labels, gap)
            score_perm = self.cross_validate(brain_map, labels_perm)
            scores_perms[j] = score_perm
            if score_perm > base_score:
                count += 1

        return ((count + 1) / (self.n_perm + 1)), scores_perms

    def classify(self, brain_maps, labels, do_pval=True):
        """ Attention, this function is based on labels with consecutive, balanced categories, like['U','U','D','D',
        'R','R','L','L']
        :param brain_maps: list (for 1 subject) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param do_pval: boolean to tell if it is needed to estimate a p-value
        :return: cross-validation score, p-value """

        cv_score = self.cross_validate(brain_maps[0], labels)
        p_val, scores_perm = None, None
        if do_pval:
            p_val, scores_perm = self.p_value_random_permutations(brain_maps[0], labels, cv_score)
        return cv_score, p_val, scores_perm

    def classify_tasks_regions(self, brain_maps, labels, tasks_names, regions_names, do_pval=True):
        """
        :param brain_maps: list (size n_subjects) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param tasks_names: list of strings corresponding to the different experiments
        :param regions_names: list of strings corresponding to the different masks to use
        :param do_pval: boolean to tell if it is needed to estimate a p-value
        :return: One dictionary with cross-validation scores for the different experiment-mask combo,
                and another dictionary with the corresponding p-values """
        cv_scores = dict()
        p_values = dict()
        scores_perm = dict()
        for task_name in tasks_names:
            for region_name in regions_names:
                _maps = [maps[region_name] for maps in brain_maps[task_name]]
                key = task_name + "_" + region_name
                cv_scores[key], p_values[key], scores_perm[key] = self.classify(_maps, labels[task_name], do_pval)

        return cv_scores, p_values, scores_perm

    def within_modality_decoding(self, maps, labels, subjects_ids, tasks_regions):
        within_cv_scores = [dict() for _ in subjects_ids]
        within_p_values = [dict() for _ in subjects_ids]
        for i, subj_id in enumerate(subjects_ids):
            # within-modality decoding : training on a task and decoding on other samples from same task
            for tasks, regions in tasks_regions:
                cv_sc, p_val, scores_perm = self.classify_tasks_regions(maps[i], labels, tasks, regions, do_pval=False)
                within_cv_scores[i].update(cv_sc)
                within_p_values[i].update(p_val)
            # print("Within-modality decoding done for subject "+str(subj_id)+"/"+str(n_subjects))

        return within_cv_scores

    def unary_cross_modal_decoding(self, brain_maps, labels, tasks_names, regions_names):
        """
        :param brain_maps: list (size n_subjects) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param tasks_names: list of strings corresponding to the different experiments
        :param regions_names: list of strings corresponding to the different masks to use
        :return: a dictionary with the test scores obtained when fitting on one experiment and testing on another,
                for the different brain regions """

        def sub(task_order):
            scores = dict()
            key = "cross_"
            for region_name in regions_names:
                _maps_0 = [maps[region_name] for maps in brain_maps[task_order[0]]]
                _maps_1 = [maps[region_name] for maps in brain_maps[task_order[1]]]
                for i in range(len(_maps_0)):
                    # across voxels demeaning
                    scaler = StandardScaler(with_std=False)
                    map_0 = (scaler.fit_transform(_maps_0[i].T)).T
                    map_1 = (scaler.fit_transform(_maps_1[i].T)).T

                    self.model.fit(map_0, labels[task_order[0]])
                    predictions = self.model.predict(map_1)
                    self.update_predictions(predictions)
                    acc = self.model.score(map_1, labels[task_order[1]])
                    scores[key + region_name] = acc
            return scores

        scores_0 = sub(tasks_names)
        scores_1 = sub(tasks_names[::-1])

        return average_dicos([scores_0, scores_1])

    def cross_modality_decoding(self, maps, labels, subjects_ids, tasks_regions):
        cross_cv_scores = [dict() for _ in subjects_ids]
        cross_p_values = [dict() for _ in subjects_ids]
        for i, subj_id in enumerate(subjects_ids):
            # cross-modal decoding : training on a task and decoding on samples from another task
            for tasks, regions in tasks_regions:
                scores_cross_mod = self.unary_cross_modal_decoding(maps[i], labels, tasks, regions)
                cross_cv_scores[i].update(scores_cross_mod)
            # print("Cross-modal decoding done for subject "+str(subj_id)+"/"+str(n_subjects))

        return cross_cv_scores

    def produce_permuted_labels(self, labels, n_perm):
        """
        This function produces n_perm labels datasets, which are obtained by permuting the original labels
        :param labels: labels to shuffle
        :param n_perm:
        :return:
        """
        gap = int(len(labels) / self.n_classes)
        perm_labels = [None] * n_perm
        for i in range(n_perm):
            perm_labels[i] = permute_labels(labels, gap)

        return perm_labels

    def score_bootstrapped_permutations(self, n_single_perm, labels, tasks_regions, maps, n_subjects, within_modality):
        start_time = time.time()
        scores_n_perm = [None]*n_subjects
        for i in range(n_subjects):
            labels_shuffled_vis = self.produce_permuted_labels(labels, n_single_perm) # repeating for each subject, such that we don't obtain same permutations
            labels_shuffled_aud = self.produce_permuted_labels(labels, n_single_perm)
            scores_dicts = [dict() for _ in range(n_single_perm)]
            for j in range(n_single_perm) :
                labels_dico = {"vis" : labels_shuffled_vis[j], "aud" : labels_shuffled_aud[j]}
                for task_regions in tasks_regions :
                    tasks, regions = task_regions
                    if within_modality :
                        cv_sc, _, _ = self.classify_tasks_regions(maps[i], labels_dico, tasks, regions, do_pval=False)
                    else :
                        cv_sc = self.cross_modal_decoding(maps[i], labels_dico, tasks, regions)
                    scores_dicts[j].update(cv_sc)

            scores_n_perm[i] = scores_dicts

        duration = time.time()-start_time
        print("Running models done in "+str(duration)+" seconds")
        return scores_n_perm
