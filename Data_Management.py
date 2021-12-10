import items as items
import train as train
from matplotlib import collections
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import pandas as pd
from collections import OrderedDict

filepath = "D:\Travail\Project_Data\MatlabData/"


def load_save_data(filepath):
    filename = "trainFD001"
    file = filepath + filename + ".mat"
    mat = loadmat(file)
    raw_data = mat[filename]
    train_set = {}
    for j in range(int(np.max(raw_data[:, 0]))):
        train_set["motor" + str(j)] = raw_data[raw_data[:, 0] == j + 1, :]
        train_set["motor" + str(j)] = train_set["motor" + str(j)][:, 1:]
    savemat("trainset.mat", train_set)


def find_constant(trainset, acc=10 ** -8, sensorsName=None):
    mean1 = np.mean(trainset["motor0"], 0)
    count = np.zeros([1, mean1.shape[0]])
    for keys, value in trainset.items():
        if keys[:3] == "mot":
            for k in range(value.shape[0]):
                count += (value[k, :] - mean1) ** 2
    count = count > acc
    for keys, value in trainset.items():
        if keys[:3] == "mot":
            m, n = value.shape
            n = int(np.sum(count))
            mask = np.ones([m, 1]) * count
            trainset[keys] = np.reshape(value[mask == 1], (m, n))
    if sensorsName is not None:
        sensorName.drop(np.nonzero(count == 0)[1], axis=0)
    return trainset, sensorsName


def normalize_batch(trainset):
    # define min max scaler
    sca = MinMaxScaler()
    # transform data
    for keys, value in trainset.items():
        if keys[:3] == "mot":
            value = sca.fit_transform(value)
            value[:, [0, 1, 2, 9]] = 0
            trainset[keys] = value
    return trainset


def pca_batch(data):
    pca = PCA(svd_solver='full', tol=0.0, iterated_power='auto', random_state=None)
    for keys, value in data.items():
        if keys[:3] == "mot":
            value = pca.fit_transform(value)
            data[keys] = value[:, :5]
    return data


def denoise(data):
    h = np.ones([5,1])/5
    for k, value in data.items():
        if k[:3] == "mot":
            for i in range(value.shape[1]):
                print(value.shape)
                data[k][:, i] = signal.convolve(value[:, i], h, 'same')
    return data


################################# Loading and saving of the dataset ###########################
sensorsName = ["n_flight",
               "atm0",
               "atm1",
               "atm2",
               "T2",
               "T24",
               "T30",
               "T50",
               "P2",
               "P15",
               "P30",
               "Nf",
               "Nc",
               "epr",
               "Ps30",
               "phi",
               "NRf",
               "NRc",
               "BPR",
               "farB",
               "htBleed",
               "Nf_dmd",
               "PCNfR_dmd",
               "W31",
               "W32"]
load_save_data(filepath)
sensorsName = pd.DataFrame(sensorsName)
sensorsName.to_csv(r'D:\Travail\Project_Data\sensorsName.csv', index=False, sep=';')
################################################################################################
trainset = loadmat("trainset.mat")
sensorName = pd.read_csv("sensorsName.csv", sep=';')
# ################################# Processing data ##############################################
# Normalize the data
trainset = normalize_batch(trainset)
# Remove the 0 columns
trainset, sensorsName = find_constant(trainset, 10 ** -8, sensorName)
savemat("trainset.mat", trainset)
# sensorsName.to_csv(r'D:\Travail\Project_Data\sensorsName.csv', index=False, sep=';')
# # Extract the baseline
# test = denoise(trainset)
# savemat("test.mat", test)
# # # PCA
pca = pca_batch(trainset)
# # # Save the modified .mat and the name of the sensors we kept
savemat("pca.mat", pca)
# np.save("sensorName", sensorsName)
################################# Processing data ##############################################
traindata = loadmat("pca")
#
# motor1 = train["motor1"]
#
# plt.plot(pca)
# plt.show()
#
