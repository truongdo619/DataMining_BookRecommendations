class EvalHelper(object):

    def __init__(self, hr_top, model = 'NCF'):
        self.top_n = hr_top
        self.model = model
    
    def getResultFromKNN(self, inputs):
        ### to do here
        # return top_n book_id having highest score
        return [1]
    
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
