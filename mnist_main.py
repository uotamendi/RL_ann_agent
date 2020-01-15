from hyperparameters import Hyperparameters
from mnistDataloader import MnistDataloader
from mnistCnn import MnistCnn
import random

hyperparameters=Hyperparameters()

dataLoader=MnistDataloader()
dataLoader.process_data()
dataLoader.convert_to_binary_class()
##Generate random hyperparameters

activations=["elu","relu","tanh","selu","sigmoid","exponential"]
layer1_neurons=random.randint(16,64)
aux_kernel=random.randint(2,5)
layer1_kernel=(aux_kernel,aux_kernel)
layer1_act=activations[random.randint(0,5)]

layer2_neurons=random.randint(32,128)
aux_kernel=random.randint(2,3)
layer2_act=activations[random.randint(0,5)]
layer2_dropout=random.uniform(0,0.6)
layer2_kernel=(aux_kernel,aux_kernel)


aux_pool=random.randint(2,3)
pool_size=(aux_pool,aux_pool)

dense_neurons=random.randint(32,128)
dense_act=activations[random.randint(0,5)]
dense_dropout=layer2_dropout=random.uniform(0,0.6)

learning_rate=random.uniform(0,1.5)

epochs= random.randint(1,5)

batch_size=random.randint(5,25)

hyperparameters.add_list([["layer1_neurons",layer1_neurons],["layer1_kernel",layer1_kernel],["layer1_act",layer1_act],
["layer2_neurons",layer2_neurons],["layer2_act",layer2_act],
["layer2_dropout",layer2_dropout],["layer2_kernel",layer2_kernel],
["pool_size",pool_size],["dense_neurons",dense_neurons],
["dense_dropout",dense_dropout],["dense_act",dense_act],
["learning_rate",learning_rate],["epochs",epochs],
["batch_size",batch_size],
])

network=MnistCnn(dataLoader,hyperparameters)
network.loadModel()
network.fit()