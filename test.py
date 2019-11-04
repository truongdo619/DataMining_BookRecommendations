import csv
import itertools
from EvalNCF import EvaluationNCF

eval = EvaluationNCF(10)

inputs = [153, 1, [(112, []), (274, []), (341, []), (42, [])]]

print(eval.query(inputs))

# with open('dataset/books.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)

#     with open('dataset/abc.csv', 'w', newline='') as cb:
#         writer = csv.writer(cb, delimiter=',',
#                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         writer.writerow(["book_id", "title"])
#         tmp = []
#         for item in itertools.islice(reader, 1000):
#             writer.writerow([item['book_id'], item['title']])

