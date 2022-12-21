from Classifier import KNearestNeighbours
import json
from operator import itemgetter
import test


with open(r'data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)

with open(r'titles.json', 'r+', encoding='utf-8') as f:
    titles = json.load(f)


target = [0 for item in titles]

test_point = test.AVENGERS_INFINITY_WAR

model = KNearestNeighbours(data, target, test_point, k=10)

model.fit()

max_distances = sorted(model.distances, key=itemgetter(0))[-1]

for i in model.indices:
    print(titles[i][0] + ' -----> ' + str(model.distances[i][0]))