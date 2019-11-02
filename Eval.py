from EvalHelper import EvalHelper
from utils import *

class Eval(object):

    def __init__(self, HR_top, model):
        self.top_n = HR_top
        self.model = model
        self.eval_helper = EvalHelper(self.top_n, self.model)

    def eval_user(self, user_id):
        user_em, rm_book_id, books_to_eval = leaveOneOutUser(user_id)
        # feed to model to get top n books
        top_n_books_id = self.eval_helper.getResult((user_em, books_to_eval))
        if rm_book_id in top_n_books_id:
            return 1 # it's a hit
        return 0

def runEval(HR_top, model):
    eval = Eval(HR_top, model)
    hits_count = 0
    
    for user_id in range(1, nb_user):
        hits_count = hits_count + eval.eval_user(user_id)

    HR_score = hits_count / nb_user
    return HR_score
    ### to do run multiple times for better evaluation

if __name__ == "__main__":
    ### to do add some configure here
    HR_top = 10
    model = 'NCF'
    print(runEval(HR_top, model)) # later needed to edit here, save to text or csv for human read
    