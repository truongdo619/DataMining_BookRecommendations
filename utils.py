
import numpy as np
from helper.load_data_helper import load_csv_to_numpy

data = load_csv_to_numpy("dataset/new_interactions.csv")
nb_user = np.amax(np.delete(data,1,1))
nb_item = np.amax(np.delete(data,0,1))

cache_user_embeddings = {}
cache_item_embeddings = {}

def getUserEmbedding(userID):
   if userID in cache_user_embeddings.keys():
      return np.array(cache_user_embeddings[userID])
   items = np.zeros(nb_item, dtype=np.int8)
   tmp = np.delete(data[data[:, 0] == userID], 0 , 1).flatten()
   for item in tmp:
      items[item - 1] = 1
   cache_user_embeddings[userID] = items;
   return np.array(cache_user_embeddings[userID])

def getItemEmbedding(itemID):
   if itemID in cache_item_embeddings.keys():
      return np.array(cache_item_embeddings[itemID])
   users = np.zeros(nb_user, dtype=np.int8)
   tmp = np.delete(data[data[:, 1] == itemID], 1 , 1).flatten()
   for item in tmp:
      users[item - 1] = 1
   cache_item_embeddings[itemID] = users;
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
   
   assert(len(obs_books) > 0)
   np.random.shuffle(obs_books)
   rm_book_id = obs_books[0]
   userEm[rm_book_id - 1] = 0
   rm_book_em = getItemEmbedding(rm_book_id)
   rm_book_em[userID - 1] = 0

   books = [(rm_book_id, rm_book_em)]
   for book_id in unobs_books:
      books.append((book_id, getItemEmbedding(book_id)))
   return (userEm, rm_book_id, books)


   
   
