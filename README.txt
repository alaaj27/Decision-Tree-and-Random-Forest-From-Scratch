Team : RuntimeTerror
Members: Ala Jararweh
Title: Decision Trees and Random Forest

1. Important Notes: 
	- I will provide a Jupyter notebook, if you prefer working on notebooks.
	- I will provide separate files if you prefer working on command line.
	- test.csv and train.csv will not be submitted. You need to add them manually.


2. Dependencies :
	install numpy  	— For math functions.
	install pandas 	— For using DataFrame functionality.
	install sklearn  — We use model_selection.train_test_split to split the dataset into training and validation.
	install scipy	— We use scipy.stats to look-up chi table.
	install collections.Counter


3. How to run:

	1st step: store train.csv and test.csv in the same directory where main.py exists.
	2nd step: run python main.py


4. File Descriptions:

	- main.py: runs the classifier using training data. 
	- utils.py: contains some python functions to facilitate data cleaning and feature extraction.
	- Dtree.py: responsible for building the decision tree only.
	- DecisionTreeClassifier.py : responsible for sampling the dataset and start the training process.


5. Functions Description:

a) Dtree class:
	- build : building the tree with respect to hyper-parameters and statical tests.
	- predict: predict a set of rows at once.
	- predictOne: predict one row at a time.
	- MakeSplitDecision: Based on a specific error metric, this function returns the feature with the highest information gain.
	- chi_square: Given a feature, return True if that feature useful for performing the next split. return False, if not.
	- CalculateIG: calculate information gain for a given feature
	- GiniIndex, MCE, Entropy: Implementations for the impurity metrics 


b) DecisionTreeClassifier class:
	- fit: start the training process
	- predict: call predict methods in Dtree to predict set of rows.
	- accuracy_score: calculate the accuracy score for a given X and y.
	- ensemble: responsible for sampling and bagging. If n_ensembles parameter is greater than 1, this function will be called to create a forest.


c) Utils file:
	- replace : given a string return a new a string with a char replaced randomly from a given list.
	- split: Split a string into chunks based on chunk_size
	- n_grams :Split the samples into features of grams based into chunk_size
	- preprocess : for preprocessing steps including replace ambiguous letters, generate features based on n-gram model, and dealing with duplicate rows.

