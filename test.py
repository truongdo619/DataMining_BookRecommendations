import csv
import itertools

with open('dataset/books.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    with open('dataset/abc.csv', 'w', newline='') as cb:
        writer = csv.writer(cb, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["book_id", "title"])
        tmp = []
        for item in itertools.islice(reader, 1000):
            writer.writerow([item['book_id'], item['title']])