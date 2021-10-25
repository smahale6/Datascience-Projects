import BagLearner as bag
import LinRegLearner as lr
import numpy as np
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.bag_learners = list()
        for i in range(20):
            self.bag_learners.append(bag.BagLearner(learner=lr.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def author(self):
        return 'Shrikanth Mahale'
    def add_evidence(self, xTrain, yTrain):
        for bag in self.bag_learners:
            bag.add_evidence(xTrain, yTrain)
    def query(self, xTest):
        predictions = list()
        for bag in self.bag_learners:
            prediction = bag.query(xTest)
            predictions.append(prediction)
        predictions = np.mean(np.asarray(predictions), axis=0)
        return predictions