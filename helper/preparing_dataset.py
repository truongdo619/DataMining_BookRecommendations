import csv
from numpy import genfromtxt,savetxt
import numpy as np
from load_data_helper import load_csv_to_numpy
import random

def preparing(path):
    data = load_csv_to_numpy(path)
    nb_line = data.shape[0]
    nb_user = np.amax(np.delete(data, 1, 1))
    nb_item = np.amax(np.delete(data, 0, 1))
    arr = np.zeros(shape=(nb_line,2), dtype=np.int16)
    count = 0
    while count < 100:
        rand_user = random.randrange(1,nb_user+1)
        rand_item = random.randrange(1,nb_item+1)
        tmp = [rand_user, rand_item]
        if not (data == np.array(tmp)).all(1).any() and not (arr == np.array(tmp)).all(1).any():
            arr[count] = tmp
            count += 1
            print (count)
    paths = path.split("/")
    paths[len(paths) - 1] = "no" + paths[len(paths) - 1]
    path = "/".join(paths)
    savetxt(path, arr, delimiter="\t", fmt="%d")


if __name__ == '__main__':
    preparing("dataset/small_dataset/interactions_small.csv")
    preparing("dataset/small_dataset/interactions_6M.csv")