import csv
from numpy import genfromtxt,savetxt
import numpy as np
from load_data_helper import load_csv_to_numpy
import random

def preparing(path, threshold = 20):
    data = load_csv_to_numpy(path)
    nb_user = np.amax(np.delete(data, 1, 1))
    arr = [ item if data[data[:, 0] == item].shape[0] > threshold else 0 for item in range(1, nb_user+1)]
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

def poprow(my_array,pr):
    i = pr
    pop = my_array[i].reshape(1,2)
    new_array = np.vstack((my_array[:i],my_array[i+1:]))
    return (new_array, pop)

def generate_test_interaction(path):
    data = load_csv_to_numpy(path)
    nb_user = np.amax(np.delete(data, 1, 1))
    tmp = data[data[:, 0] == 1]
    nb_row = tmp.shape[0]
    rand = random.randrange(nb_row)
    tmp = poprow(tmp, rand)
    train_interactions, test_interactions = tmp
    for item in range(2, nb_user+1):
        tmp = data[data[:, 0] == item]
        nb_row = tmp.shape[0]
        rand = random.randrange(nb_row)
        tmp = poprow(tmp, rand)
        train_interactions = np.append(train_interactions, tmp[0], axis=0)
        test_interactions = np.append(test_interactions, tmp[1], axis=0)
    savetxt("dataset/interactions.csv", train_interactions, delimiter="\t", fmt="%d")
    savetxt("dataset/test_interactions.csv", test_interactions, delimiter="\t", fmt="%d")

# if __name__ == '__main__':
    # generate_test_interaction("dataset/interactions.csv")