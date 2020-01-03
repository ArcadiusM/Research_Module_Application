import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def extractFromNames(train, test):
    """  Extract title and length from name without the words in brackets. Add dummy for name with brackets. """
    for df in [train, test]:
        df['Name_Len'] = df['Name'].str.replace(r"\(.+\)", "").apply(lambda x: len(x))
        df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        df['Name_Correction'] = df['Name'].str.contains(r'\(.+\)', regex=True).astype(int)
        del df['Name']
    return train, test


def imputeAge(train, test):
    """ Fill NaN for age with the respective Name_Title and Pclass mean. """
    for df in [train, test]:
        df['Age_Null_Flag'] = df['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    train['mean'] = train.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')
    train['Age'] = train['Age'].fillna(train['mean'])
    merged_data = test.merge(train, on=['Name_Title', 'Pclass'], how='left').drop_duplicates(['PassengerId_x'])
    test['Age'] = np.where(test['Age'].isnull(), merged_data['mean'], test['Age'])
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    del train['mean']
    return train, test


def extractCabinLetter(train, test):
    """ Extract first letter of cabin. """
    for df in [train, test]:
        df['Cabin_Letter'] = df['Cabin'].apply(lambda x: str(x)[0])
        del df['Cabin']
    return train, test


def imputeEmbarked(train, test):
    """ Fill NaN for Embarked by 'S'. """
    for df in [train, test]:
        df['Embarked'] = df['Embarked'].fillna('S')
    return train, test


def createDummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Cabin_Letter', 'Name_Title']):
    """ Convert specified columns into dummy variables.

    Parmeters:
        train(pandas.DataFrame): Training data
        test(pandas.DataFrame): Test data
        columns(list): Set of columns to be converted

    Returns:
        train(pandas.DataFrame): Training data containing dummies
        test(pandas.DataFrame): Test data containing dummies
    """
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column + '_' + col for col in train[column].unique() if col in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


def extractFamilySize(train, test):
    """ Categorize the size of the family into the three categories. """
    for df in [train, test]:
        df['Fam_Size'] = np.where((df['SibSp'] + df['Parch']) == 0 , 'Solo',
                           np.where((df['SibSp'] + df['Parch']) <= 3,'Nuclear', 'Big'))
        del df['SibSp']
        del df['Parch']
    return train, test

