import csv
from numpy import genfromtxt,savetxt
import numpy as np

def load_csv_to_numpy(path):
   my_data = genfromtxt(path, delimiter='\t',dtype=int,skip_header=0)
   # savetxt("dataset/interactions.csv", my_data[((my_data[:, 1] <= 1000) & (my_data[:, 0] <= 1000 ))], delimiter="\t", fmt="%d")
   return my_data

# data = load_csv_to_numpy("dataset/interactions_6M.csv")