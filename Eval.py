from EvalHelper import EvalHelper
from utils import *
import argparse, yaml

class Eval(object):

    def __init__(self, HR_top, model):
        self.top_n = HR_top
        self.model = model
        self.eval_helper = EvalHelper(self.top_n, self.model)
        with open("config.yml", 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)

    def eval_user(self, user_id):
        user_em, rm_book_id, books_to_eval = leaveOneOutUser(user_id)
        # feed to model to get top n books
        top_n_books_id = self.eval_helper.getResult((user_id, user_em, books_to_eval))
        print(user_id, top_n_books_id)
        if rm_book_id in top_n_books_id:
            return 1
        return 0

def runEval(HR_top, model):
    eval = Eval(HR_top, model)
    hits_count = 0
    
    for user_id in range(1, nb_user + 1):
        hits_count = hits_count + eval.eval_user(user_id)

    HR_score = hits_count / nb_user
    return HR_score
    ### to do run multiple times for better evaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='NCF',
                        help='input data path')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n book recommendations')
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = parse_args()
    HR_top = args.top_n
    model = args.model
    print(runEval(HR_top, model)) # later needed to edit here, save to text or csv for human read
    