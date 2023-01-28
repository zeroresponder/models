from numpy import loadtxt
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# load the dataset
# Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium,Heart Disease
dataset = loadtxt('Heart_Disease_Prediction.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:13]
y = dataset[:,13]

print(X[0])

def create_model():
    model = Sequential()
    model.add(Dense(16, input_shape=(13,), activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=150, batch_size=10)
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save("model.h5")
    return model

def load_model():
    model = keras.models.load_model("model.h5")
    return model

model = load_model()
print(model.predict(numpy.asarray([X[0]])))
