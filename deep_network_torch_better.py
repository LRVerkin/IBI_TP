# -*- coding: utf-8 -*-
import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
from torch.autograd import *
from torch.nn import *
import math



def affichage(image,label):
    # on récupère à quel chiffre cela correspond (position du 1 dans label)
    label = numpy.argmax(label)
    # on crée une figure
    plt.figure()
    # affichage du chiffre
    # le paramètre interpolation='nearest' force python à afficher chaque valeur de la matrice sans l'interpoler avec ses voisines
    # le paramètre cmap définit l'échelle de couleur utilisée (ici noire et blanc)
    plt.imshow(image,interpolation='nearest',cmap='binary')
    # on met un titre
    plt.title('chiffre '+str(label))
    # on affichage les figures créées
    plt.show()




# nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
TRAIN_BATCH_SIZE = 1
# on charge les données de la base MNIST
data = pickle.load(gzip.open('mnist_light_CN.pkl.gz'),encoding='latin1')
# images de la base d'apprentissage
train_data = torch.Tensor(data[0][0])
# labels de la base d'apprentissage
train_data_label = torch.Tensor(data[0][1])
# images de la base de test
test_data = torch.Tensor(data[1][0])
# labels de la base de test
test_data_label = torch.Tensor(data[1][1])
# on crée la base de données d'apprentissage (pour torch)
train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
# on crée la base de données de test (pour torch)
test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
# on crée le lecteur de la base de données d'apprentissage (pour torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# on crée le lecteur de la base de données de test (pour torch)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
# 10 fois





class Deep_Model:

    def __init__(self,nb_hidden_layers,size_hidden_layers,activation_function):
        
        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.activation_function = activation_function
        # D_in is input dimension; D_out is output dimension.
        self.D_in, self.D_out = 785, 10
        self.model = self.create_model()



    def create_model(self):
        
        model = torch.nn.ModuleList([torch.nn.Linear(self.D_in, self.size_hidden_layers)])
    
        if activation_function == "Sigmoid":
            for i in range(self.nb_hidden_layers):
                model.append(torch.nn.Sigmoid())
        if activation_function == "RELu":
            for i in range(self.nb_hidden_layers):
                model.append(torch.nn.RELU())
        if activation_function == "Tanh":
            for i in range(self.nb_hidden_layers):
                model.append(torch.nn.Tanh())

        model.append(torch.nn.Linear(self.size_hidden_layers, self.D_out))
        return model

    def forward(self,x):
        # forward pass from input data x
        # through all of the model
        # returns: prediction of the model for x

        y = self.model[0](x)
        for i in range(1,len(self.model)):
            y = self.model[i](y)

        return y

    def train_model(self,images_train, T_train,learning_rate):
        
        t = 0

        # Loss function used: MSE
        loss_fn = torch.nn.MSELoss(reduction='sum')

        for image,label in images_train:
            image = numpy.transpose(numpy.insert(image[0,:].numpy().flatten(), 0, 1))
            image = image.reshape((785,))
            x = torch.from_numpy(image)
            label.resize_(10)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.forward(x)


            # Compute loss
            loss = loss_fn(y_pred, label)
            #print(t, loss.item())

            # Zero the gradients of all layers in ModuleList before running the backward pass.
            for module in self.model:
                module.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model.
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

            t += 1
            if t > T_train:
                break

        return model


    def use_model(self,images_test, T_test):
    # T_test is testing set size
    # images_set is testing set of images

        t = 0
        nbOK = 0

        for image, label in images_test:
            
            image = numpy.transpose(numpy.insert(image[0,:].numpy().flatten(), 0, 1))
            image = image.reshape((785,))
            x = torch.from_numpy(image)
            label.resize_(10)

            y_pred = self.forward(x)

            value_pred, index_pred = y_pred.max(0)
            value_true, index_true = label.max(0)
            if index_true.item()==index_pred.item():
                nbOK += 1
            #print("Real value is "+str(index_true.item())+" -- prediction is "+str(index_pred.item()))
            t += 1
            if t > T_test:
                break

        print("OK ratio: "+str(1.0*nbOK/T_test))




activation_function = ""
while activation_function not in ["Sigmoid","RELU","Tanh"]:
    activation_function = input("Activation Function for the network: Sigmoid, RELU or Tanh?\n")

model = Deep_Model(nb_hidden_layers = 2,size_hidden_layers = 30,activation_function=activation_function)
model.create_model()
model.train_model(images_train = train_loader, T_train = 10000, learning_rate = 1e-2)
model.use_model(images_test = test_loader, T_test = 300)