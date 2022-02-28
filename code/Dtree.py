
import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats  # For looking up values in chi-square  table



class Dtree :
  def __init__(self ,
               X = None ,
               y = None ,
               depth = 0,
               MAX_DEPTH = 3,
               error = "entropy",
               features = None ,
               label = None,
               feature = None,
               Type = None,
               AllFeatures = [],
               value =None,
               NextSplitBy = None,
               confidenceLevel = 0.05,
               minSampleSize = 20
               ):

        self.children = {}
        self.X = X
        self.y = y
        self.MAX_DEPTH = MAX_DEPTH
        self.AllFeatures = AllFeatures
        self.NextSplitBy = NextSplitBy
        self.feature = feature
        self.value = value
        self.label = label
        self.Type = Type

        self.confidenceLevel = confidenceLevel
        self.depth = depth
        self.error = error
        self.minSampleSize = minSampleSize

        self.Bulid()


  def Bulid (self):

    if self.depth >= self.MAX_DEPTH:
        self.label = max (dict (self.y.value_counts()))
        self.Type = "leaf"
        return



    if len (self.y.unique()) == 1: #one class left in Y
      self.label = max (dict (self.y.value_counts()))
      self.Type = "leaf"
      return


    if len(self.AllFeatures) == 0: # No features left
      self.label = max (dict (self.y.value_counts()))
      self.Type = "leaf"
      return


    if len (self.y) < self.minSampleSize :   #min sample examples
      self.Type = "leaf"
      self.label = max (dict (self.y.value_counts()))
      return



    self.NextSplitBy = self.MakeSplitDecision(self.AllFeatures , self.error)

    if self.NextSplitBy is None:
      self.AllFeatures = []
      self.label = self.y.unique()[0]
      self.Type ="leaf"
      return

    if not (self.chi_square(self.NextSplitBy , self.X , self.y)):
        self.AllFeatures =[]
        return

    if len(self.y.unique()) > 1 :

        possibleValues = self.X[self.NextSplitBy].unique()

        for v in possibleValues:

          SubsetX = self.X[self.X[self.NextSplitBy] == v] # return rows where feature == specific values
          SubsetY = self.y[SubsetX.index]

          m = [f for f in self.AllFeatures if f != self.NextSplitBy]

          child = Dtree(X = SubsetX.copy() ,
                        y = SubsetY.copy() ,
                        error = self.error,
                        depth = self.depth +1 ,
                        label = max (dict (SubsetY.value_counts())),
                        AllFeatures = m,
                        MAX_DEPTH = self.MAX_DEPTH,
                        feature = self.NextSplitBy,
                        value = v,
                        Type = "internal",
                        confidenceLevel = self.confidenceLevel,
                        minSampleSize = self.minSampleSize
                        )

          self.children[v]= child # append to children

        for v in possibleValues:
          self.children[v].Bulid()




    return self


  def predict (self , df):
    """ recieves a dataframe and returns a list of predictions """


    predictions = []
    setOfFeature= set()
    for indx in range(len(df)):

        pred = self.predictOne(dict(df.iloc[indx]).copy())

        predictions.append(pred)

    return predictions

  def predictOne(self , row) -> str:


      node = self
      label = "Undefined"

      features = None #self.AllFeatures

      while (node is not None ):

        if len(node.children) != 0 :

              branchName = row[node.NextSplitBy]
              child =None
              try :
                child = node.children[branchName]
              except:
                return node.label

              row.pop (node.NextSplitBy)

              node = child

        else:
              label = node.label
              break

      return label



  def MakeSplitDecision (self , features , error):
    """
    Perform spliting on the tree until:
        1- the maximum depth is reached,
        2- one class remains, or
        3- A performance metric is achieved.
    """

    #determine the best split based on the features

    BestFeature = None
    BestInformationGain = -1
    BestFeatureList = []

    #consider all error metrics for the split. For a list of features,
    #we take the feature that satisfies most of error metrics

    if error == "all" or error == "All":
      for err in {"Entropy" , "MCE" , "gini"}:
        BestFeatureList = {}

        for f in features:

          IG = self.CalculateIG(f , err)

          if BestInformationGain < IG:
            BestFeature = f
            BestInformationGain = IG

        BestFeatureList[err] = BestFeature

      return max(Counter(BestFeatureList.values())) #return the feature suggested by most error metrics

    else:
      for f in features:

          IG = self.CalculateIG(f , error)

          if BestInformationGain < IG:
            BestFeature = f
            BestInformationGain = IG

      return BestFeature


  def chi_square(self, feature , X , y  ):

    """
    Given a feature, return True if that feature useful for performing the next split.
    return False, if not.
    """
    if self.confidenceLevel == 0: #always expand
        return True

    possibleValues = set(X[feature])

    X_counts = dict (X[feature].value_counts())

    y_counts = dict (Counter(y))


    Sum = 0
    for v in possibleValues:
      SubsetX = X[X[feature] == v]
      SubsetY = y[SubsetX.index]

      SubsetY_counts = dict (Counter(SubsetY))


      for key in y_counts.keys():
        actual = SubsetY_counts.get(key , 0)
        expected = len(SubsetY) * ((y_counts.get(key , 0) / len(y)))

        Sum += ((actual - expected)**2)/expected


    dFreedom = (len(y_counts) - 1) * (len(possibleValues) - 1)



    chi_value = scipy.stats.chi2.ppf(1- self.confidenceLevel, df=dFreedom)


    if (Sum > chi_value):
      return True

    return False

  def CalculateIG(self , feature , error ):
    """
      loop over feature's values to calculate the maximum Information Gain.
    """

    #Get unique values for that attributes:
    Xtemp = self.X.copy()
    ytemp = self.y.copy()

    PossibleValuesForFeature= Xtemp[feature].unique()

    IG = None

    if error == "MCE" or error == "mce":
      IG = self.MCE(Xtemp, ytemp)
    elif error == "entropy" or error == "Entropy":
      IG = self.Entropy(Xtemp, ytemp)
    elif error == "gini" or error == "Gini":
      IG = self.GiniIndex(Xtemp, ytemp)
    else:
      print("enter a valid error metric name , {MCE, entropy, Gini, or  All}.")



    for value in PossibleValuesForFeature:
      SubsetX = Xtemp[Xtemp[feature] == value] # return rows where feature == specific values
      SubsetY = ytemp [SubsetX.index]

      IG -=  (SubsetX.shape[0] / Xtemp.shape[0]) * self.Entropy(SubsetX, SubsetY)
      if error == "MCE" or error == "mce":
         IG -=  (SubsetX.shape[0] / Xtemp.shape[0])  * self.MCE(SubsetX, SubsetY)


      elif error == "entropy" or error == "Entropy":
        IG -= (SubsetX.shape[0] / Xtemp.shape[0]) * self.Entropy(SubsetX, SubsetY)


      elif error == "gini" or error == "Gini":
        IG -=  (SubsetX.shape[0] / Xtemp.shape[0]) * self.GiniIndex(SubsetX, SubsetY)


      else:
        print("enter a valid error metric name , {MCE, entropy, Gini, or  All}.")

    return IG



  def GiniIndex(self, Xtemp, ytemp ):

    gini = 0

    for c in ytemp.unique(): #returns the rows where class == c
      splitCandidate = Xtemp[ytemp[:] == c]

      L1 = len(splitCandidate)
      L2 = len(Xtemp)

      ratio = L1 / L2
      gini += (ratio ** 2)

    return 1- gini




  def Entropy(self, XTemp , yTemp):
    """
    For attribute x_i, calculate the value of entropy
    """

    entropy = 0
    for c in yTemp.unique(): #returns the rows where class == c
      splitCandidate= XTemp[yTemp[:] == c]

      L1 = len(splitCandidate)
      L2 = len(XTemp)

      ratio = L1 / L2
      entropy += ratio * np.log2(ratio)


    return -1 * entropy #take every class i and do (- p_i .log p_i)





  def MCE(self , Xtemp, ytemp):
    """
     - Miss-Classification Error(MCE): measures the amount
     of samples that classified with an incorrect label.

     - calculating the MCE for each attributes (A) then
     the attribute with minimum MCE is selected at each node.
    """
    mceList = []

    for c in ytemp.unique(): #returns the rows where class == c
      splitCandidate = Xtemp[ytemp[:] == c]

      L1 = len(splitCandidate)
      L2 = len(Xtemp)

      ratio = L1 / L2
      mceList.append(ratio)

    return 1- max (mceList)
