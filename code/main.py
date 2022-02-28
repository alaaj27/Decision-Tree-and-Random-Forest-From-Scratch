
from sklearn.model_selection import train_test_split    #Split the dataset into training and validation
from utils import *
from DecisionTreeClassifier import *
import os



def main():

    #import the training dataset from the current working directory
    dataset = pd.read_csv(os.getcwd() + '/train.csv' , names = ["id", "sample", "label"])

    #preprocess the training dataset
    X , y = preprocess(dataset, ngrams= 1 , yFlag=True, duplicates="drop",unknown = "no")

    #import and preprocessing the test dataset

    # UNCOMMENT THE NEXT TWO LINES and LINE 43, if you want to work with the test.csv dataset
    #testData = pd.read_csv(os.getcwd()+"/test.csv" , names = ["id", "sample"])
    #X_test , _ = preprocess(testData, ngrams= 1 , yFlag=False, duplicates="no", unknown = "no")

    #split the data into training and validation
    X_train, X_CV, y_train, y_CV = train_test_split(X, y, test_size = 0.2,random_state = 0)


    #Set-up the best hyper-parameters,
    #Dataset bagging will be performed when n_ensembles > 1

    print("\nTraining started ...")
    print("Training might take a while to complete!")
    classifier = DecisionTreeClassifier(error_metric="gini" ,
                                        minimumSampleSize =4 ,
                                        MAX_DEPTH = 7,
                                        confidence_level= 0.999,
                                        n_ensembles=2)

    #train the model
    classifier.fit(X_train,y_train)

    #final accuracy scores by all models together.
    print("- Train: ",classifier.accuracy_score(X_train,y_train))
    print("- CV: "  , classifier.accuracy_score(X_CV, y_CV))
    #print("Test: " , classifier.accuracy_score(X_test, y_True["class"]))





if __name__ == "__main__":
    main()
