from utils import *

num_users_test = 1000

for i in range(1, num_users_test):
    a = leaveOneOut_user(i)[0]
    b = getUserEmbedding(i)
    if(sum(b - a) != 1):
        print(sum(b - a))
    assert(sum(b - a) == 1)

