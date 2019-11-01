import csv
from numpy import genfromtxt,savetxt

def load_csv_to_numpy(path):
   my_data = genfromtxt(path, delimiter='\t',dtype=int,skip_header=0)
   # savetxt("ratings.csv", my_data[: , :-1], delimiter="\t", fmt="%d")
   return my_data