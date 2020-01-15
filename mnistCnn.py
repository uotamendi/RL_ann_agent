from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


class MnistCnn:
    def __init__(self,dataLoader,hyperparameters):
        self.dataLoader= dataLoader
        self.hyperparameters=hyperparameters
        self.model=None


    def loadModel(self):
        
        layer1_neurons=self.hyperparameters.get("layer1_neurons")
        layer1_act=self.hyperparameters.get("layer1_act")
        layer1_kernel=self.hyperparameters.get("layer1_kernel")

        layer2_neurons=self.hyperparameters.get("layer2_neurons")
        layer2_act=self.hyperparameters.get("layer2_act")
        layer2_dropout=self.hyperparameters.get("layer2_dropout")
        layer2_kernel=self.hyperparameters.get("layer2_kernel")

        pool_size=self.hyperparameters.get("pool_size")

        dense_neurons=self.hyperparameters.get("dense_neurons")
        dense_act=self.hyperparameters.get("dense_act")
        dense_dropout=self.hyperparameters.get("dense_dropout")

        model = Sequential()
        model.add(Conv2D(layer1_neurons, kernel_size=layer1_kernel,
                        activation=layer1_act,
                        input_shape=self.dataLoader.input_shape))
        model.add(Conv2D(layer2_neurons, layer2_kernel, activation=layer2_act))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(layer2_dropout))
        model.add(Flatten())
        model.add(Dense(dense_neurons, activation=dense_act))
        model.add(Dropout(dense_dropout))
        model.add(Dense(self.dataLoader.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(learning_rate=self.hyperparameters.get("learning_rate")),
                    metrics=['accuracy'])


        self.model=model

    def fit(self):

        self.model.fit(self.dataLoader.x_train, self.dataLoader.y_train,
                batch_size=self.hyperparameters.get("batch_size"),
                epochs=self.hyperparameters.get("epochs"),
                verbose=1,
                validation_data=(self.dataLoader.x_test, self.dataLoader.y_test))
        score = self.model.evaluate(self.dataLoader.x_test, self.dataLoader.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])