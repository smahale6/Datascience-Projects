from scipy import stats
import RTLearner as rt
import numpy as np

class BagLearner(object):

    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = list()
        bag = 0
        while bag < bags:
            self.learners.append(learner(**self.kwargs))
            bag += 1
        if verbose == True:
            self.fetch_learner_info()

def author():
    return 'Shrikanth Mahale'

    def add_evidence(self, trainX, trainY):
        bag = 0
        samples = trainX.shape[0]
        while bag < self.bags:
            rand_sample  = np.random.choice(samples, samples)
            Bag_X = trainX[rand_sample]
            Bag_Y = trainY[rand_sample]
            self.learners[bag].add_evidence(Bag_X, Bag_Y)
            bag += 1

    def query(self, TestX):
        predictions = list()
        bag = 0
        while bag < self.bags:
            prediction = self.learners[bag].query(TestX)
            predictions.append(prediction)
            bag += 1
        predictions = np.asarray(predictions)
        predictions = stats.mode(predictions, axis=0)
        return predictions[0]

    def fetch_learner_info(self):
        print("bags =", self.bags)
        print("kwargs =", self.kwargs)
        print("boost =", self.boost)



        