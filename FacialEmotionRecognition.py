#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:40:28 2022

@author: tomscholer & amitjadhav
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sn
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers, models, metrics, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D, Bidirectional, ConvLSTM2D, LSTM
import pickle

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

import atexit

#function to split the data in the respective training, validation subsets
def split(data):
    train=data[data.iloc[:, 2] == 'Training']
    valid=data[data.iloc[:, 2] == 'PrivateTest']
    test=data[data.iloc[:, 2] == 'PublicTest']
    return train, valid, test

#function to split the data into a set of images (pixels) and a set of results (emotions)
def build_data(data, number_of_classes, shape=(48,48)):
    X,Y=[],[]
    for i in range(data.shape[0]):
        d=data.iloc[i,:]
        image=d['pixels']
        image=np.fromstring(image, dtype='uint8', sep=' ')
        image=image.reshape(shape)
        X.append(image)
        y=np.zeros(number_of_classes)
        y[d['emotion']]=1
        Y.append(y)
    return np.array(np.expand_dims(X,-1)),np.array(Y)

def feature_map_layer(l):
    submodel = models.Model(inputs=model.inputs, outputs=model.layers[l].output)
    submodel.summary()
    print(X.shape)
    fm = submodel.predict(X[8:9, :, :, None])

    fig, axs = plt.subplots(8, 8)
    fig.set_size_inches(10, 10)

    c = 0
    for i in range(8):
        for j in range(int((model.layers[l].filters)/8)):
            axs[j][i].imshow(fm[0, :, :, c], cmap='gray')
            c += 1
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])
            if (j==7):
              break
    plt.savefig('weights_visualisation_layer_{}_{}.png'.format(l,model_name))
    plt.show()

def regularize(data):
    return data/255


# Automatically clear memory on forceful closure to prevent hogging
@atexit.register
def on_exit():
    tf.keras.backend.clear_session()
    
    
# Build model here
def build_model(num_outputs, input_shape,final_model_name, kernel_size=3):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(name=final_model_name)
    if final_model_name == 'Model_1':
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(num_outputs))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_2':
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(num_outputs))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_3':
        tf.random.set_seed(50)
        model.add(layers.Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(7, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_4':
        tf.random.set_seed(50)
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(num_outputs))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_5_BatchNorm':
        tf.random.set_seed(50)
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(7))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_6':
        tf.random.set_seed(50)
        model.add(layers.Conv2D(16, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(7, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    elif final_model_name == 'Model_7_Extra_Conv_TanhSigmoid':
        tf.random.set_seed(50)
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='tanh'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='sigmoid', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(7))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    elif final_model_name == 'Model_8_New':
        model.add(layers.InputLayer(input_shape=input_shape))
        

    print(model.summary())
    return model


print('Decompressing the dataset...')
data = pd.read_csv(sys.argv[1])

print(data.head())
print(f'Data has the shape {data.shape}')


number_of_emotions=len(np.unique(data['emotion']))
print(f'Number of emotion classes: {number_of_emotions}')

number_of_emotions=len(np.unique(data['emotion']))
print(f'Number of emotion classes: {number_of_emotions}')    

t, v, test = split(data)
print('Training shape: {}, Validation shape: {}'.format(t.shape, v.shape))
X, Y = build_data(t, number_of_emotions)
X = regularize(X)
gen = ImageDataGenerator( rotation_range = 15, width_shift_range = 0.15, height_shift_range = 0.15, shear_range = 0.15, zoom_range = 0.15, horizontal_flip = True)
# gen = ImageDataGenerator( rotation_range = 15, width_shift_range = 0.25, height_shift_range = 0.25, brightness_range=[0.8,1.2], shear_range = 0.15, zoom_range = [0.8,1.2], horizontal_flip = True)
gen.fit(X)
print(X.shape, '\n', Y.shape)



X_valid, Y_valid = build_data(v, number_of_emotions)
X_valid = regularize(X_valid)
print('X shape: {}, Y shape: {}'.format(X.shape, Y.shape))
print('X Validatn shape: {}, Y Validatn shape: {}'.format(X_valid.shape, Y_valid.shape))
final_model_name = 'Model_4'#'Model_7_Extra_Conv_TanhSigmoid'#'Model_5_BatchNorm'#'Model_7_Mine'#'Model_3_OTHERS'#'Model_4'#'Model_1'
model = build_model(number_of_emotions, (X.shape[1], X.shape[2], 1),final_model_name,3)

callback = callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights = True)
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, min_delta=0.0001, patience=5, verbose=1)
window_sizes = [3, 5, 7, 9, 13, 15].reverse() #this is unused
batch= 32
history = model.fit(gen.flow(X, Y, batch_size = batch), validation_data=(X_valid, Y_valid),
                    callbacks=[lr_reduce, callback],steps_per_epoch = len(X)/batch, epochs=200,validation_steps = len(X_valid)/batch, shuffle=True)



#Save model and its architecture

model_name = model.name
# Save the trained model
model.save('{}'.format(final_model_name))
#Dump the history object as dictionary
with open('{}/{}_history.pkl'.format(final_model_name, final_model_name), 'wb') as f:
    pickle.dump(history.history, f)
    

#model_name = ''
#model = tf.keras.models.load_model(model_name)

plot_model(model, to_file='{}.png'.format(model_name), show_shapes=True)

#Load history
#history = pickle.load(open('{}/history.pkl'.format(model_name),'rb'))

#plotting accuracy and loss of validation and training set over a certain amount of epochs with loaded history
plt.plot(history['loss'],label='training loss')
plt.plot(history['val_loss'],label='validation loss')
plt.legend()
plt.savefig('loss_{}.png'.format(model_name))
plt.show()
plt.plot(history['accuracy'],label='training accuracy')
plt.plot(history['val_accuracy'],label='validation accuracy')
plt.legend()
plt.savefig('accuracy_{}.png'.format(model_name))
plt.show()


#Test model on testing data set
X, Y = build_data(test, number_of_emotions)
X = regularize(X)
predictions = model.predict(X[:, :, :, None])
print(predictions.shape)
predictions_to_display = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fig, axs = plt.subplots(3, len(predictions_to_display))
fig.set_size_inches(35, 10)
for i, e in enumerate(predictions_to_display):
    axs[0][i].imshow(np.squeeze(X[e]), cmap='gray')
    axs[0][i].grid(False)
    axs[1][i].bar(range(number_of_emotions), predictions[e])
    axs[1][i].set_ylim(ymin=0)
    axs[1][i].set_title(f'Predicted emotion for image {e}')

    axs[2][i].bar(range(number_of_emotions), Y[e], color='green')
    axs[2][i].set_title(f'Actual emotion for image {e}')
    axs[2][i].set_xlabel(f'Emotion classes')
plt.savefig('predictions_{}.png'.format(model_name))
plt.show()



# Calculate precision and recall of test set

X, Y = build_data(test, number_of_emotions)
X = regularize(X)
# print(X.shape)

# predict probabilities for test set
yhat_probs = model.predict(X[:, :, :, None])
# print(yhat_probs.shape)
# predict crisp classes for test set
# yhat_classes = model.predict_classes(X[:, :, :, None], verbose=0)
yhat_classes = np.argmax(model.predict(X[:, :, :, None]), axis=-1)
# print(yhat_classes.shape)
# print(yhat_classes)
Y = np.argmax(Y, axis=1)
# print(Y.shape)
# print(Y)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y, yhat_classes)
print('Applying Model on Test set:')
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y, yhat_classes, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y, yhat_classes, average='macro')
print('Recall: %f' % recall)



# confusion matrix
matrix = confusion_matrix(Y, yhat_classes)
print(matrix)
df_cm = pd.DataFrame(matrix, index=[i for i in "0123456"],
                     columns=[i for i in "0123456"])
plt.figure(figsize=(10, 7))
svm = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
figure = svm.get_figure()
figure.savefig('svm_conf_{}.png'.format(model_name), dpi=400)



# Plot Recall, precision, accuracy
fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
values = np.array([accuracy, precision, recall]).reshape(1, -1)
print(values)
df = pd.DataFrame(values, columns=['Accuracy', 'Precision', 'Recall'])
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plt.savefig('value_matrix_{}.png'.format(model_name))
plt.show()


























    
