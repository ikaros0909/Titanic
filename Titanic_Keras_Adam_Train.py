'''
Created on 2018. 2. 12.

@author: danny
'''
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
import Titanic_Keras_Adam_Inc as ufc

# loss: 0.3444 - acc: 0.8507
# loss: 0.3247 - acc: 0.8597
# loss: 0.2786 - acc: 0.8822

train_data_read = ufc.read_data("train")
train_data_read['Sex'] = train_data_read['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train_data_read["Title"] = train_data_read["Name"].apply(lambda s:s[s.find(",")+2:s.find(".")])
train_data_read["Title"] = train_data_read.apply(ufc.assign_title, axis=1)
# print(train_data_read.groupby(["Title"]).agg(["count"]))

train_data_read["Age"].fillna(train_data_read["Age"].mean(), inplace=True) #나이는 평균입력
train_data_read["Age"] = train_data_read.apply(ufc.assign_age, axis=1) #연령대로 변경

train_data_read["FamilySize"] = train_data_read.apply(ufc.assign_familysize, axis=1) #가족크기
# print(train_data_read.groupby(["FamilySize"]).agg(["count"]))
train_data_read["Cabin"] = train_data_read["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train_data_read["Fare"] = train_data_read.apply(ufc.assign_Fare, axis=1)

train_data_read["Embarked"].fillna("S", inplace=True) #가장 일반적인 코드
train_dummies = ufc.assign_dummies(train_data_read)

# pd.concat(objs,  # Series, DataFrame, Panel object
#          axis=0,  # 0: 위+아래로 합치기, 1: 왼쪽+오른쪽으로 합치기
#          join='outer', # 'outer': 합집합(union), 'inner': 교집합(intersection)
#          join_axes=None, # axis=1 일 경우 특정 DataFrame의 index를 그대로 이용하려면 입력
#          ignore_index=False,  # False: 기존 index 유지, True: 기존 index 무시
#          keys=None, # 계층적 index 사용하려면 keys 튜플 입력
#          levels=None,
#          names=None, # index의 이름 부여하려면 names 튜플 입력
#          verify_integrity=False, # True: index 중복 확인
#          copy=True) # 복사
train_data_input = pd.concat([train_data_read, train_dummies], axis=1, join='outer', ignore_index=False)
train_data = ufc.assign_inputdata(train_data_input)
# print(train_data.describe())
# train_data["Child"] = train_data["Age"].apply(lambda s: 1 if s < 13 or s > 60 else 0)
# train_data = train_data.drop(["Age"], axis=1)

# print(train_data_read.shape)
# print(train_dummies.shape)
# print(train_data_input.shape)
# print(train_data.shape)

train_data.fillna(0, inplace=True)

# X = np.array(train_data.ix[:, 1:])
X = np.array(train_data)
# X = np.array(train_data_read["Sex"].apply(lambda s: 1 if s == 'female' else 0))
y = np.ravel(train_data_read.Survived)

#model for a binary classification to predict wine type
model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_shape=(X.shape[1],)))
# model.add(Dense(X.shape[1]*2, activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
# model.add(Dense(X.shape[1]*2, activation='relu'))
model.add(Dense(X.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
model.fit(X, y, epochs=1000, batch_size=1, verbose=2) #verbose=2로 설정하여 progress bar 가 나오지 않도록 설정한다.


model.save("./model/titanic.model")


