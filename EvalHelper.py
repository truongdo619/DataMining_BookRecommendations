from EvalKNN import KnnRecommender
from EvalNCF import EvaluationNCF

class EvalHelper(object):

    def __init__(self, hr_top, model='NCF'):
        self.top_n = hr_top
        self.model = model
        self.recommenderKNN = KnnRecommender()
        self.recommenderNCF = EvaluationNCF(self.top_n)

    def getResultFromKNN(self, inputs):
        # return top_n book_id having highest score
        return self.recommenderKNN.make_recommendations(inputs, self.top_n)
    
    def getResultFromNCF(self, inputs):
        # return top_n book_id having highest score
        return self.recommenderNCF.query(inputs)

    def getResult(self, inputs):
        # input : (user_id, user_em, [(book_id, book_em)])
        # output: top_n book_id that have highest score, break tie randomly
        if self.model == 'KNN':
            return self.getResultFromKNN(inputs)
        elif self.model == 'NCF':
            return self.getResultFromNCF(inputs)
