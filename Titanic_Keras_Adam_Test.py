'''
Created on 2018. 2. 12.

@author: danny
'''
import numpy as np 
import pandas as pd 
from keras.models import load_model, Sequential
from keras.layers import Dense 
import Titanic_Keras_Adam_Inc as ufc

test_data_org = ufc.read_data("test")
test_data_read = test_data_org
test_data_read['Sex'] = test_data_read['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_data_read["Title"] = test_data_read["Name"].apply(lambda s:s[s.find(",")+2:s.find(".")])
test_data_read["Title"] = test_data_read.apply(ufc.assign_title, axis=1)
test_data_read["Age"].fillna(test_data_read["Age"].mean(), inplace=True) #나이는 평균입력
test_data_read["Age"] = test_data_read.apply(ufc.assign_age, axis=1) #연령대로 변경
test_data_read["FamilySize"] = test_data_read.apply(ufc.assign_familysize, axis=1) #가족크기
test_data_read["Cabin"] = test_data_read["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data_read["Fare"] = test_data_read.apply(ufc.assign_Fare, axis=1)
test_data_read["Embarked"].fillna("S", inplace=True) #가장 일반적인 코드
test_dummies = ufc.assign_dummies(test_data_read)
test_data_input = pd.concat([test_data_read, test_dummies], axis=1, join='outer', ignore_index=False)
test_data = ufc.assign_inputdata(test_data_input)

test_data.fillna(0, inplace=True)
X_test = np.array(test_data)

model = Sequential()
model.add(Dense(X_test.shape[1], activation='relu', input_shape=(X_test.shape[1],)))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
# model.add(Dense(X_test.shape[1]*2, activation='relu'))
model.add(Dense(X_test.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

model = load_model("./model/titanic.model")
predictions = np.round(model.predict(X_test))
# predictions = np.array(predictions, dtype=np.int64)
df_predictions = pd.DataFrame(data=predictions)
test_result = pd.concat([test_data_org["PassengerId"], df_predictions.astype(int)], axis=1, join='outer', ignore_index=False)
# test_result = pd.concat([test_data_org["PassengerId"], test_data_org["Sex"].apply(lambda s: 1 if s == 'female' else 0)], axis=1, join='outer', ignore_index=False)
test_result = test_result.rename(index=str,columns={"PassengerId":"PassengerId"
                                        ,0:"Survived"
                                        })
test_result.to_csv('./data/test_result_keras_adam.csv', index=False)

