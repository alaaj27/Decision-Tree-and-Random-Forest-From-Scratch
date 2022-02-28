# Decision Trees and Random Forest implementation in python

We provide a python implementation for Decision Tree and Random Forest in python. We use different splitting criteria such as Information gain with Entropy, GiniIndex, and Miss-Classification Error. Chi_square test is performed on every feature to determine the importance of that node for the tree. Moreover, we use other stopping criteria metrics such as MAX_DEPTH and minimumSampleSize of the tree to avoid overfitting.

# Problem Description
[The link](https://www.kaggle.com/c/cs529project1-2022/overview) for the competition on Kaggle.

Splice junctions are points on a DNA sequence at which superfluous DNA is removed during the process of protein creation in higher organisms. The problem posed in this dataset is to recognise, given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks: recognising exon/intron boundaries (referred to as EI sites), and recognising intron/exon boundaries (IE sites). (In the biological community, IE borders are referred to as ""acceptors"" while EI borders are referred to as ""donors"".)


# Dependencies :
	`install numpy`  	— For math functions.
	`install pandas` 	— For using DataFrame functionality.
	`install sklearn`  — We use model_selection.train_test_split to split the dataset into training and validation.
	`install scipy`	— We use scipy.stats to look-up chi table.
	`install collections.Counter`


# How to Run:
	- run `python main.py` on your command line
	- For random Forest, change the attribute n_ensembles to value grater than one and run.

# File Descriptions:

	- main.py: runs the classifier using training data.
	- utils.py: contains some python functions to facilitate data cleaning and feature extraction.
	- Dtree.py: responsible for building the decision tree only.
	- DecisionTreeClassifier.py : responsible for sampling the dataset and start the training process.


# Functions Description:

## Dtree class:
	- build : building the tree with respect to hyper-parameters and statical tests.
	- predict: predict a set of rows at once.
	- predictOne: predict one row at a time.
	- MakeSplitDecision: Based on a specific error metric, this function returns the feature with the highest information gain.
	- chi_square: Given a feature, return True if that feature useful for performing the next split. return False, if not.
	- CalculateIG: calculate information gain for a given feature
	- GiniIndex, MCE, Entropy: Implementations for the impurity metrics


## DecisionTreeClassifier class:
	- fit: start the training process
	- predict: call predict methods in Dtree to predict set of rows.
	- accuracy_score: calculate the accuracy score for a given X and y.
	- ensemble: responsible for sampling and bagging. If n_ensembles parameter is greater than 1, this function will be called to create a forest.


## Utils file:
	- replace : given a string return a new a string with a char replaced randomly from a given list.
	- split: Split a string into chunks based on chunk_size
	- n_grams :Split the samples into features of grams based into chunk_size
	- preprocess : for preprocessing steps including replace ambiguous letters, generate features based on n-gram model, and dealing with duplicate rows.
