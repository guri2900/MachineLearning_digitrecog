#This is a modified version of load_all_data from the first homoework adapted to work with the digit recognition data.
#The test and train files must be placed in the 'data' folder in the same directory as this file.
import numpy as np


def load_all_data():
    #Use numpy to load "train.csv" an dskip the first row
    train = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1)
    test = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1)

    train_labels = train[:,0]
    train_data = train[:,1:]
    test_data = test[:,1:]
    test_labels = test[:,0]

    return train_data, test_data, train_labels, test_labels
