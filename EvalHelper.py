from KNN_recommender import KnnRecommender

class EvalHelper(object):

    def __init__(self, hr_top, model='NCF'):
        self.top_n = hr_top
        self.model = model
        self.recommender = KnnRecommender()

    def getResultFromKNN(self, inputs):
        # return top_n book_id having highest score
        return self.recommender.make_recommendations(inputs, self.top_n)
    
    def getResultFromNCF(self, inputs):
        ### to to here
        # return top_n book_id having highest score
        return [1]

    def getResult(self, inputs):
        # input : (user_em, [(book_id, book_em)])
        # output: top_n book_id that have highest score, break tie randomly
        if self.model == 'KNN':
            return self.getResultFromKNN(inputs)
        elif self.model == 'NCF':
            return self.getResultFromNCF(inputs)
