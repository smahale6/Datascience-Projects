import DTLearner as dt
import numpy as np

class BagLearner(object):

    def __init__(self, learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        bag = 0
        while bag < bags:
            self.learners.append(learner(**self.kwargs))
            bag += 1
        if verbose == True:
            self.fetch_learner_info()

    def author(self):
        return 'Shrikanth Mahale'

    def add_evidence(self, xTrain, yTrain):
        bag = 0
        samples = xTrain.shape[0]
        while bag < self.bags:
            rand_sample  = np.random.choice(samples, samples)
            Bag_X = xTrain[rand_sample]
            Bag_Y = yTrain[rand_sample]
            self.learners[bag].add_evidence(Bag_X, Bag_Y)
            bag += 1

    def query(self, xTest):
        predictions = []
        bag = 0
        while bag < self.bags:
            prediction = self.learners[bag].query(xTest)
            predictions.append(prediction)
            bag += 1
        predictions = np.asarray(predictions)
        predictions = np.mean(predictions, axis=0)
        return predictions

    def fetch_learner_info(self):
        print("bags =", self.bags)
        print("kwargs =", self.kwargs)
        print("boost =", self.boost)



        