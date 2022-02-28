from Dtree import *
import pandas as pd
import numpy as np

class DecisionTreeClassifier:

  def __init__(self, error_metric= "entropy",
               confidence_level = 0.05 ,
               MAX_DEPTH = 3,
               n_ensembles = 1,
               stopping_critera = "Chi_square",
               minimumSampleSize = 20) :

    self.forest = []
    self.MAX_DEPTH = MAX_DEPTH
    self.stopping_critera = stopping_critera
    self.error_metric = error_metric
    self.confidence_level = confidence_level
    self.minimumSampleSize = minimumSampleSize
    self.n_ensembles = n_ensembles

  def fit (self, X_temp, y_temp):

    if self.n_ensembles == 1 :
        Tree = Dtree (X = X_temp ,
                       y= y_temp,
                       AllFeatures= X_temp.columns,
                       Type = "root",
                       MAX_DEPTH = self.MAX_DEPTH,
                       error = self.error_metric ,
                       confidenceLevel = self.confidence_level,
                       minSampleSize = self.minimumSampleSize)

        self.forest.append(Tree)

    elif self.n_ensembles > 1:
        self.ensemble(X_temp, y_temp)

    else:
        raise ValueError('Error n_ensembles cannot be negative!')


  def predict (self, X):

    ids = X.index
    predictions = {i:{"IE":0 ,"EI":0, "N":0 } for i in ids}

    for i in ids:  #iterate over all rows

        #predict row_i using all models
        for tree in self.forest :

            # predict one row using tree_i
            label = tree.predictOne(dict(X.loc[i]))

            # increase the predicted label counter
            predictions[i][label] +=1

    #extract labels with max predicted count
    forestPredictions = [max(predictions[i],
                             key=predictions[i].get) for i in predictions.keys()]

    return np.array(forestPredictions)



  def accuracy_score(self, X_test , y_true):
        y_pred = self.predict(X_test)
        return np.sum(np.equal(y_true, y_pred)) / len(y_true)


  def ensemble (self, X, y):


    #sampled rows should be attached with the correct label.
    # We need to sample both X and y together.
    df = X.copy()
    df["label"] = y

    # reset index to start from zero, pd.sample method takes ordered dataframe
    df = df.reset_index(drop=True)


    for i in range (self.n_ensembles):

        df_sampled = df.sample(frac=1 , axis= 0 , replace=True).reset_index(drop=True)

        y_sampled = df_sampled["label"]
        X_sampled = df_sampled.drop(labels='label', axis=1)


        Tree = Dtree (X = X_sampled,
                       y= y_sampled,
                       AllFeatures= X_sampled.columns,
                       Type = "root",
                       MAX_DEPTH = self.MAX_DEPTH,
                       error = self.error_metric ,
                       confidenceLevel = self.confidence_level,
                       minSampleSize = self.minimumSampleSize)

        self.forest.append(Tree)
        print("Classifier #", i+1, "completed training: ")
        print("\tAccuracy on Training data: ", self.accuracy_score(X_sampled, y_sampled))
        print()
