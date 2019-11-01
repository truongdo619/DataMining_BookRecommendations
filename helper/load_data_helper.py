import csv
from numpy import genfromtxt,savetxt
import numpy as np

def load_csv_to_numpy(path):
   my_data = genfromtxt(path, delimiter='\t',dtype=int,skip_header=0)
   # savetxt("ratings_small.csv", my_data[((my_data[:, 1] <= 2000) & (my_data[:, 0] <= 5000 ))], delimiter="\t", fmt="%d")
   return my_data

# data = load_csv_to_numpy("dataset/small_dataset/interactions_small.csv")