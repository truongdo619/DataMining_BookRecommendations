
import numpy as np
from helper.load_data_helper import load_csv_to_numpy
import yaml
import random


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

data = load_csv_to_numpy(cfg["data_path"]["small"]) # (user_id, book_id)
nb_user = np.amax(np.delete(data,1,1))
nb_item = np.amax(np.delete(data,0,1))

tmp = load_csv_to_numpy(cfg["data_test_path"])
user_to_remove_bookID = { item[0] : item[1] for item in tmp}

cache_user_embeddings = {}
cache_item_embeddings = {}

def getUserEmbedding(userID):
   if userID in cache_user_embeddings.keys():
      return np.array(cache_user_embeddings[userID])
   items = np.zeros(nb_item, dtype=np.int8)
   tmp = np.delete(data[data[:, 0] == userID], 0 , 1).flatten()
   for item in tmp:
      items[item - 1] = 1
   cache_user_embeddings[userID] = items
   return np.array(cache_user_embeddings[userID])

def getItemEmbedding(itemID):
   if itemID in cache_item_embeddings.keys():
      return np.array(cache_item_embeddings[itemID])
   users = np.zeros(nb_user, dtype=np.int8)
   tmp = np.delete(data[data[:, 1] == itemID], 1 , 1).flatten()
   for item in tmp:
      users[item - 1] = 1
   cache_item_embeddings[itemID] = users
   return np.array(cache_item_embeddings[itemID])

# return userEmbedding after remove one interaction and list of unobserved book embeddings
# (user_em, remove_book_id, [(book_id, book_em)])
def leaveOneOutUser(userID):
   userEm = getUserEmbedding(userID)
   unobs_books, obs_books = [], []
   for book_id in range(1, nb_item):
      if userEm[book_id - 1] == 0:
         unobs_books.append(book_id)
      else:
         obs_books.append(book_id)
         
   rm_book_id = user_to_remove_bookID[userID]
   rm_book_em = getItemEmbedding(rm_book_id)

   books = [(rm_book_id, rm_book_em)]
   unobs_books = random.sample(unobs_books, cfg['random_sample_test_KNN'])
   for book_id in unobs_books:
      books.append((book_id, getItemEmbedding(book_id)))

   return (userEm, rm_book_id, books)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    res = []
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            res.append(shuffled_data[start_index:end_index])
    return res
    