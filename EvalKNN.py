from utils import getUserEmbedding, getItemEmbedding

import os
import numpy as np
import random
import statistics
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

np.seterr(divide='ignore', invalid='ignore')

class KnnRecommender:
    def __init__(self):
        print('Loading')
        print('......\n')

    def make_recommendations(self, inputs, top_n=10):
        user_id = inputs[0]
        user_em = inputs[1]
        books = inputs[2]
        indices = [i for i in range(1, user_em.shape[0] + 1) if user_em[i - 1] != 0]
        sample = indices
        cos_dict = {}
        for book in books:
            book_id, book_em = book
            tmp = [self.cos_sim(getItemEmbedding(item), book_em) for item in sample]
            cos_dict[book_id] = statistics.mean(tmp)
        return dict(sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]).keys()

    @staticmethod
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

# if __name__ == '__main__':
    # recommender = KnnRecommender()
    # user_em = getUserEmbedding(1)
    # indices = [i for i in range(1, user_em.shape[0] + 1) if user_em[i - 1] == 0]
    # books = []
    # for i in indices:
    #     books.append((i, getItemEmbedding(i)))
    # recommender.make_recommendations((user_em, books))
