import numpy as np
from operator import itemgetter

class KNearestNeighbours:

    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k
        self.distances = list()
        self.categories = list()
        self.indices = list()
        self.counts = list()
        self.category_assigned = None

    @staticmethod
    def manhattan(p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))

    # def dist(p1, p2):
        # return np.linalg.norm(np.array(p1) - np.array(p2))


    def fit(self):
        self.distances.extend([(self.manhattan(self.test_point, point), i) for point, i in zip(self.data, [i for i in range(len(self.data))])])

        sorted_list = sorted(self.distances, key=itemgetter(0))

        self.indices.extend([index for (val, index) in sorted_list[:self.k]])

        for i in self.indices:
            self.categories.append(self.target[i])

        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])

        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]