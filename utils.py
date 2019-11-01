
import numpy as np
from helper.load_data_helper import load_csv_to_numpy

data = load_csv_to_numpy("ratings.csv")
nb_user = np.amax(np.delete(data,1,1))
nb_item = np.amax(np.delete(data,0,1))

def getUserEmbedding(userID):
   items = np.zeros(nb_item, dtype=np.int8)
   tmp = np.delete(data[data[:, 0] == userID], 0 , 1).flatten()
   for item in tmp:
      items[item - 1] = 1
   return items

def getItemEmbedding(itemID):
   users = np.zeros(nb_user, dtype=np.int8)
   tmp = np.delete(data[data[:, 1] == itemID], 1 , 1).flatten()
   for item in tmp:
      users[item - 1] = 1
   return users

# print(getItemEmbedding(258))