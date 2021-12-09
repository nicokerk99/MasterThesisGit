import random


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
        :return: the estimated p-value for classification on this data """

        random.seed(self.seed)
        count = 0
        gap = int(len(labels) / self.n_classes)
        for _ in range(self.n_perm):
            labels_copy = labels.copy()

            # shuffling but keeping a structure
            for i in range(gap):
                indexes = range(i,len(labels),gap)
                subset_copy = [labels_copy[idx] for idx in indexes]
                random.shuffle(subset_copy)
                labels_copy[i:len(labels):gap] = subset_copy

            score_perm = self.cross_validate(brain_map, labels_copy)
            if score_perm > base_score:
                count += 1

        return count / self.n_perm

    def classify(self, brain_maps, labels):
        """ Attention, this function is based on labels with consecutive, balanced categories, like['U','U','D','D',
        'R','R','L','L']
        :param brain_maps: list (size n_subjects) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :return: list (size n_subjects) cross-validation scores, and corresponding p-values """

        n_subjects = len(brain_maps)
        cv_scores = []
        p_values = []

        for i in range(n_subjects):
            score = self.cross_validate(brain_maps[i], labels)
            cv_scores.append(score)
            p_values.append(self.p_value_random_permutations(brain_maps[i], labels, score))

        return cv_scores, p_values

    def classify_tasks_regions(self, brain_maps, labels, tasks_names, regions_names):
        """
        :param brain_maps: list (size n_subjects) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param tasks_names: list of strings corresponding to the different experiments
        :param regions_names: list of strings corresponding to the different masks to use
        :return: One dictionary with cross-validation scores for the different experiment-mask combo,
                and another dictionary with the corresponding p-values """
        cv_scores = dict()
        p_values = dict()
        for task_name in tasks_names:
            for region_name in regions_names:
                _maps = [maps[region_name] for maps in brain_maps[task_name]]
                key = task_name + "_" + region_name
                cv_scores[key], p_values[key] = self.classify(_maps, labels[task_name])

        return cv_scores, p_values

    def cross_modal_decoding(self, brain_maps, labels, tasks_names, regions_names):
        """
        :param brain_maps: list (size n_subjects) of lists (size n_samples) of maps, which are features
        :param labels: list of strings (size n_samples), which are the labels
        :param tasks_names: list of strings corresponding to the different experiments
        :param regions_names: list of strings corresponding to the different masks to use
        :return: a dictionary with the test scores obtained when fitting on one experiment and testing on another,
                for the different brain regions """

        scores = dict()

        def sub(task_order):
            key = task_order[0] + "_" + task_order[1] + "_"
            for region_name in regions_names:
                scores[key + region_name] = []
                _maps_0 = [maps[region_name] for maps in brain_maps[task_order[0]]]
                _maps_1 = [maps[region_name] for maps in brain_maps[task_order[1]]]
                for i in range(len(_maps_0)):
                    self.model.fit(_maps_0[i], labels[task_order[0]])
                    acc = self.model.score(_maps_1[i], labels[task_order[1]])
                    scores[key + region_name].append(acc)
            return

        sub(tasks_names)
        sub(tasks_names[::-1])

        return scores