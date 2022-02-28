import random
import pandas as pd
import numpy as np

def replace (s , replace, listOfStr):
    """
        given a string return a new a string with a char replaced randomly from a given list.
    """
    countReplace = s.count(replace)
    for i in range (s.count(replace)):
        s = s.replace(replace , random.choice(listOfStr) , 1 )

    return s


def split(s: str ,chunk_size: int ):
    """
        Split a string into chunks based on chunk_size
    """

    listStr =[]

    for i in range(0, len(s), chunk_size) :
        listStr.append(s[i:i+chunk_size])

    return listStr

def n_grams(df , chunk_size=1):
    """
        Split the samples into features of grams based into chunk_size
    """

    features = list(range(1, int( len(df["sample"][0])/chunk_size ) +1 , 1))
    List = [split(s , chunk_size) for s in df["sample"]]

    df = pd.concat(
        [df, pd.DataFrame(
            List,
            index=df.index,
            columns=features) ] , axis=1 )

    return df , features

def preprocess (df, ngrams = 1 , yFlag = False , duplicates = "no", unknown="no"):

    """
    perform the preprocessing steps:
        - replace ambiguous letters
        - generate features based on n-gram model
        - deal with duplicate rows

        parameters:
            1. y_flag: True, if df contains labels. The labels will be extracted.
            2. duplicates: if "drop", the duplicate rows will be dropped. Otherwise, they will be kept.
            3. unknown: if "drop", rows with ambiguous letters will be dropped.
                     Otherwise, they will be replaced based on "replacements" dictionary.

    """

    replecements = {"D": ["A", "G", "T"],
                "N": ["A", "G", "C", "T"],
                "S": ["C" , "G"],
                "R" : ["A" , "G"]}

    if duplicates == "drop":
        df.drop_duplicates(subset=['sample'])

    if unknown == "drop":
        unknowDNA = df["sample"].apply(lambda x: ("N" in x) or ("D" in x) or ("S" in x) or ("R" in x))
        df = df.drop( unknowDNA[unknowDNA ==True].index )
    else:
        df['sample'] = df['sample'].apply(lambda x: replace(x, 'D' , replecements["D"]))
        df['sample'] = df['sample'].apply(lambda x: replace(x, 'N' , replecements["N"]))
        df['sample'] = df['sample'].apply(lambda x: replace(x, 'S' , replecements["S"]))
        df['sample'] = df['sample'].apply(lambda x: replace(x, 'R' , replecements["R"]))


    df, features = n_grams(df , chunk_size = ngrams)
    X = df[features]
    y =None

    if yFlag:
        y = df["label"]

    return X , y
