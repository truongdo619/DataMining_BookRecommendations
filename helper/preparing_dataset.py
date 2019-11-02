import csv
from numpy import genfromtxt,savetxt
import numpy as np
from load_data_helper import load_csv_to_numpy
import random

def preparing(path, threshold = 20):
    data = load_csv_to_numpy(path)
    nb_user = np.amax(np.delete(data, 1, 1))
    arr = [ item if data[data[:, 0] == item].shape[0] > threshold else 0 for item in range(1, nb_user+1) ]
    result = None
    count = 0
    for item in range(1,nb_user+1):
        if result is None:
            result = data[data[:, 0] == item]
        elif arr[item - 1] != 0:
            print(item)
            tmp = data[data[:, 0] == item]
            for in_item in range(tmp.shape[0]):
                print(tmp[in_item, 0])
                tmp[in_item, 0] = item - count
            result = np.append(result, tmp, axis=0)
        elif arr[item - 1] == 0:
            count += 1
    savetxt("dataset/new_interactions.csv", result, delimiter="\t", fmt="%d")

if __name__ == '__main__':
    preparing("dataset/interactions.csv")