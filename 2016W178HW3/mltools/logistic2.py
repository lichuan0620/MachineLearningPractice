import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

from .base import classifier
# from .base import regressor
# from .utils import toIndex, fromIndex, to1ofK, from1ofK
# from numpy import asarray as arr
# from numpy import atleast_2d as twod
# from numpy import asmatrix as mat


################################################################################
# LOGISTIC REGRESSION CLASSIFIER ###############################################
################################################################################


class logisticClassify2(classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes :       a list of the possible class labels
        theta   :       linear parameters of the classifier (1xN numpy array, where N=# features)
        step_constant:  determine how fast the size of the steps decreases over the gradient
                        decent process; larger value means slower decrease

    """

    step_constant = 2.0

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        self.theta = np.array([])

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


    def __repr__(self):
        str_rep = 'logisticClassify2 model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


    def __str__(self):
        str_rep = 'logisticClassify2 model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


## CORE METHODS ################################################################

    def plotBoundary(self, X, Y):
        """ Plot the (linear) decision boundary of the classifier along with the data """
        assert len(self.theta) == 3, 'plotBoundary: 2d model only'
        c = self.classes[0]
        plt.plot(X[Y == c, 0], X[Y == c, 1], 'ko', color='b', label='Class 0')
        c = self.classes[1]
        plt.plot(X[Y == c, 0], X[Y == c, 1], 'ko', color='r', label='Class 1')
        axis = plt.axis()
        des_x = [axis[0], axis[1]]
        des_y = [((x*self.theta[1] + self.theta[0])/-self.theta[2]) for x in des_x]
        plt.plot(des_x, des_y, 'k-', label='Decision boundary')
        plt.axis(axis)

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        # You do not need to implement this function.
        # If you *want* to, it should return an Mx2 numpy array "P", with
        # P[:,1] = probability of class 1 = sigma( theta*X )
        # P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    def predict(self, data):
        """ Return the predictied class of each data point in X"""
        assert data.shape[1] is 2, 'Prediction requires the data to have exactly two features'
        return [self.classes[1] if self.theta[0] + np.dot(self.theta[1:], d) > 0 else self.classes[0] for d in data]

    def train(self, X, Y, init_step=1, min_change=1e-4, iteration_limit=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        # preparation
        data_point = X.shape[0]
        if Y.shape[0] != data_point:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        features = np.concatenate((np.ones((data_point, 1)), X), axis=1)
        targets = ml.toIndex(Y, self.classes)
        if self.theta.shape[0] != features.shape[1]:
            self.theta = np.random.rand(features.shape[1])

        # training
        negative_log_likely = []
        error = []
        for i in range(0, iteration_limit):
            step = self.step_constant * init_step / (self.step_constant + i)

            # gradient decent
            for j in range(0, data_point):
                # sigma = (1 + e^(-linear_response))^-1
                sigma = 1/(1 + np.exp(-np.dot(features[j], self.theta)))
                # NIL = -avg(
                #       log(sigma)      if Yi = 1
                #       log(1-sigma)    if Yi = 0
                #   )
                # d_NIL/d_theta = -avg(
                #       (1 - sigma)*Xi  if  Yi = 1
                #       -sigma*Xi       if  Yi = 0
                #   )
                gradient = (1 - sigma)*features[j] if targets[j] else -sigma*features[j]
                self.theta += step*gradient

            # record current error rate and surrogate loss
            error.append(self.err(X, Y))
            sigma = 1/(1 + np.exp(-(np.dot(features, self.theta))))
            negative_log_likely.append(-np.mean(targets*np.log(sigma)+(1-targets)*np.log(1-sigma)))

            # plot
            # TODO: this clear-and-re-plot method is slow, consider using dynamic update instead
            if plot:
                plt.figure(plot, (15, 7))
                plt.clf()
                plt.subplot(121)
                plt.plot(negative_log_likely, 'b-')
                plt.title('surrogate loss (logistic NLL)')
                plt.subplot(122)
                plt.plot(error, 'r-')
                plt.title('error rate')
                plt.draw()
                plt.pause(.01)

            # abort if there is no significant change in surrogate loss
            if (i > 1) and (abs(negative_log_likely[-1]-negative_log_likely[-2]) < min_change):
                return

################################################################################
################################################################################
################################################################################
