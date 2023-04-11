"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))

    models = []

    # TODO: INSERT YOUR CODE FOR CROSS VALIDATION HERE
    for fold in range(folds):
        # making train/test indexes
        train_i = np.ravel(np.delete(indices, fold, 1))
        test_i = indices[:,fold]
        # removing indexes==-1
        train_i = np.delete(train_i, np.where(train_i == -1))
        test_i = np.delete(test_i, np.where(test_i == -1))
        #dividing the data into train and test data
        train_d = all_data[:,train_i]
        test_d = all_data[:, test_i]
        train_l = all_labels[train_i]
        test_l = all_labels[test_i]
        
        #feeding into model
        model = trainer(train_d,train_l, params)
        models.append(model)
        predictions = predictor(test_d,model)
        
        #scoring the fold
        scores[fold] = np.mean(predictions == test_l)

    score = np.mean(scores)

    return score, models
