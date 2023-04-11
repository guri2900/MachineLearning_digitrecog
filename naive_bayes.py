"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    # creating empty model dictionary
    model = {}
    # labels = examples of class y
    labels, lab_count = np.unique(train_labels, return_counts=True)

    d, n = train_data.shape
    num_classes = labels.size

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES (USING LAPLACE ESTIMATE)

    # for storing conditional probability based on different classes, initially 0
    cond_prob = np.zeros((num_classes, d, 2))# here cols = 2 is selected because its binary classification (2 features) 
    
    #here i is a particular label
    for i in range(num_classes):
        #array of indexes where train_labels (examples) are equal to there label (unique)
        class_ind = np.where(train_labels==i)

        #storing true features
        true_fea = np.count_nonzero(
            train_data[:,class_ind[0]], 
            axis=1)
        
        # calculating conditional prob for features that are true and 
        # implementing Laplace's expansion to avoid probs. approaching 0
        cond_prob[i,:,1] = np.log((true_fea+1)/(lab_count[i]+2))
        
        # calculating cond_prob for false fetures
        cond_prob[i,:,0] = np.log((lab_count[i]-true_fea+1)/lab_count[i] + 2)
    
    # we include conditional prob of different binary features (binary classification)
    model['feature'] = cond_prob
    # also adding the type of class distribution
    model['class_distribution'] = np.log(lab_count/n)

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA
    
    # initializing the model with feature's cond_prob for true examples
    ans_true = model['feature'][:,:,1].dot(data)
    # now for false feature
    #inverting data
    rev = np.where(data == 1,0,1)
    ans_false = model['feature'][:,:,0].dot(rev)
    # summing them and resulting matrix = d X 1
    ans = ans_true+ans_false
    # taking transpose to return array like
    res = np.transpose(ans)
    final_ans = res+model['class_distribution']
    
    return np.argmax(final_ans, axis = 1)
    
    
