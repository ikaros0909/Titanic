'''
Created on 2018. 2. 13.

@author: danny
'''
import pandas as pd

# select features and labels for training
# Sex 성별
# Age 나이
# Pclass 1,2,3등석
# SibSp 함께 탐승한 형제 또는 배우자수
# Parch 함께 탑승한 부모 또는 자녀의 수
# Fare 티켓 요금
def read_data(filename):
    df = pd.read_csv("./data/%s.csv" % filename, sep=",")
    common_df = df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare", "Cabin"]]
    if filename == "train":
        return pd.concat([df["Survived"],common_df], axis=1, join='outer', ignore_index=False)
    else:
        return pd.concat([df["PassengerId"],common_df], axis=1, join='outer', ignore_index=False)

def assign_age(row):
    if row["Age"] < 13:
        return "0012"
    elif row["Age"] >= 13 and row["Age"] < 18:
        return "1317"
    elif row["Age"] >= 18 and row["Age"] < 60:
        return "1859"
    else:
        return "60Ov"

def assign_title(row):
    if row["Title"] == "Mlle" or row["Title"] == "Ms" or row["Title"] == "Mme" or row["Title"] == "Lady" or row["Title"] == "Dona":
        return "Miss"
    elif row["Title"] == "Capt" or row["Title"] == "Col" or row["Title"] == "Major" or row["Title"] == "Dr" or row["Title"] == "Rev" or row["Title"] == "Don" or row["Title"] == "Sir" or row["Title"] == "the Countess" or row["Title"] == "Jonkheer":
        return "Officer"
    else:
        return row["Title"]

def assign_Fare(row):
    if row["Fare"] < 7.91:
        return "low"
    elif row["Fare"] >= 7.91 and row["Fare"] < 14.45:
        return "middle"
    elif row["Fare"] >= 14.45 and row["Fare"] < 31:
        return "high"
    else:
        return "highest"

def assign_familysize(row):
    familysize = row["SibSp"]+row["Parch"]
    if familysize == 1:
        return "Silgle"
    elif familysize >= 2 and familysize < 5:
        return "Small"
    elif familysize >= 5:
        return "Big"
    else:
        "etc"

def assign_inputdata(df):
    return df[["Pclass_1", "Pclass_2", "Pclass_3"
               , "Sex"
               , "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Officer"
               , "Age_0012", "Age_1317", "Age_1859", "Age_60Ov"
#                , "FamilySize_Silgle", "FamilySize_Small", "FamilySize_Big"
               , "Embarked_S", "Embarked_C", "Embarked_Q"
#                , "Fare_low", "Fare_middle", "Fare_high", "Fare_highest"
               , "Cabin"
            ]]

def assign_dummies(df):
    return pd.get_dummies(df[["Pclass", "Title", "Age", "FamilySize", "Fare", "Embarked"]]
                    , prefix=['Pclass', "Title", "Age", 'FamilySize', "Fare", 'Embarked']
                    , columns=["Pclass", "Title", "Age", "FamilySize", "Fare", "Embarked"])
    
    
    