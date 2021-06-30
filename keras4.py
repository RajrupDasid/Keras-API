#predctions using Keras API and tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

test_labels=[]
test_samples=[]
for i in range(50):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    random_older=randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(1000):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    random_older=randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels=np.array(test_labels)
test_samples=np.array(test_samples)
test_labels,test_samples=shuffle(test_labels,test_samples)
scalar=MinMaxScaler(feature_range=(0,1))
scaled_test_samples=scalar.fit_transform(test_samples.reshape(-1,1))

model=Sequential([
    Dense(units=16,input_shape=(1,),activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=2,activation='softmax')
])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#adding two models
model.fit(x=scaled_test_samples,y=test_labels,validation_split=0.1,batch_size=10,epochs=30,shuffle=True,verbose=2)
# predictions
predictions=model.predict(x=scaled_test_samples,batch_size=10,verbose=0)
for i in predictions:
    print(i)
rounded_predictions=np.argmax(predictions,axis=1)
for i in rounded_predictions:
    print(i)