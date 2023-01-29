import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

scale_columns = ['age', 'chol', 'thalach']
#drop_columns = ['trestbps', "slope", "ca", 'thal', "oldpeak", 'cp', 'restecg', 'exang', 'thalach']
drop_columns = ['trestbps', "slope", "ca", 'thal', "oldpeak"]
inputs = 13 - len(drop_columns)

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
    "sex": 1,
    "cp": 0,
    "chol": 150,
    "restecg": 1,
    "fbs": 0,
    "thalach": 150,
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
    print("Mean Age:", np.mean(df['age'].to_numpy()))
    df.to_csv("heart_cleaned.csv", index=False)
    df[scale_columns] = StandardScaler().fit_transform(df[scale_columns])
    print(df.head())

    x = df.drop(['target'], axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(8, input_dim=inputs, kernel_initializer='normal',  kernel_regularizer=keras.regularizers.l1(0.001),activation='relu'))
    model.add(Dense(4, kernel_initializer='normal',  kernel_regularizer=keras.regularizers.l1(0.001),activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(x_train, y_train, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save("model.h5")
    return model


def load_model():
    model = keras.models.load_model("model.h5")
    return model

def predict(model, data):
    df = pd.read_csv("heart_cleaned.csv")
    df = pd.concat([pd.DataFrame.from_records([data]), df])

    #df = pd.DataFrame.from_records([test_data])
    print(df.head())
    df[scale_columns] = StandardScaler().fit_transform(df[scale_columns])
    print(df.head())

    input_data = df.iloc[0].to_numpy()[0:inputs]

    res = model.predict(np.asarray([input_data]))
    return res


if __name__ == "__main__":
    #create_model()
    model = load_model()

    res = predict(model, test_data)
    print(res)
