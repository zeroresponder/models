import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

inputs = 8
#scale_columns = ['age']
scale_columns = ['age', 'chol', 'thalach']
drop_columns = ['trestbps', "slope", "ca", 'thal', "oldpeak"]

# test_data = {
#     "age": 65,
#     "sex": 1,
#     "cp": 0,
#     "chol": 210,
#     "restecg": 2,
#     "fbs": 1,
#     "thalach": 179,
#     "exang": 1
# }

test_data = {
    "age": 70,
    "sex": 0,
    "cp": 1,
    "chol": 204,
    "restecg": 0,
    "fbs": 0,
    "thalach": 172,
    "exang": 0
}
#sex:
#Male = 1; Female = 0

#chest_pain_type:
#1: typical angina pain: squeezing pressur, heaviness, tightness, pain in the chest
#2: atypical angina pain: not typical
#3: non angina pain: feels like cardiac issues but isn't
#4: none

#resting_ecg:
#0: normal
#1: some abnormaliaites but could be nothing
#2: thickening in the heart, severe

#fasting_blood_sugar:
#1: fasting blood sugar is greater than 120

#max_heart_rate:
#max heart rate acheived during ecg

#exercise_induced_agnia:
#did patient experience agnia during ecg



def create_model():
    df = pd.read_csv("heart.csv")
    df = df.drop(drop_columns, axis=1)
    df.to_csv("heart_cleaned.csv", index=False)
    df[scale_columns] = StandardScaler().fit_transform(df[scale_columns])
    print(df.head())

    x = df.drop(['target'], axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(16, input_shape=(inputs,), activation='relu'))
    model.add(Dropout(.2, input_shape=(12,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save("model.h5")
    return model


def load_model():
    model = keras.models.load_model("model.h5")
    return model

if __name__ == "__main__":
    create_model()
    model = load_model()

    df = pd.read_csv("heart_cleaned.csv")
    df = pd.concat([pd.DataFrame.from_records([test_data]), df])

    #df = pd.DataFrame.from_records([test_data])
    print(df.head())
    df[scale_columns] = StandardScaler().fit_transform(df[scale_columns])
    print(df.head())

    input_data = df.iloc[0].to_numpy()[0:inputs]
    print(input_data)
    res = model.predict(np.asarray([input_data]))
    #res = model.predict(np.asarray([[1.8268750287215434,0,0,-1.8810356645907673,0,1,-1.0486919785765447,0]]))
    print(res)
