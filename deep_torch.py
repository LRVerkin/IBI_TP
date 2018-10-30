# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 08:08:38 2018

@author: Océane
"""

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
from torch.autograd import *
from torch.nn import *
import math

## implementation d'un perceptron simple

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
#for i in range(0,10):
    # on demande les prochaines données de la base
#    (_,(image,label)) = enumerate(train_loader).next()
    # on les affiche
#    affichage(image[0,:].numpy(),label[0,:].numpy())
# NB pour lire (plus proprement) toute la base (ce que vous devrez faire dans le TP) plutôt utiliser la formulation suivante


def deep(Time, w_values, eta, N_neurons,N_hidden_layers):
    cpt = 0
    cpt_blobal = 0

    for image,label in train_loader:
        image = numpy.transpose(numpy.insert(image[0,:].numpy().flatten(), 0, 1))
        image = image.reshape((785,1))
        label = numpy.transpose(label[0,:].numpy())
        label = label.reshape((10,1))
        
        T = Variable(torch.as_tensor(label), requires_grad = False) #
        X = Variable(torch.as_tensor(image), requires_grad = True)
        
        if cpt == 0:
            #initialisation
            #matrice de poids 
            losses = []
            weights = []
            weights.append(Variable(w_values*torch.randn(N_neurons, image.size), requires_grad=True))
            for i in range(N_hidden_layers-1):
                weights.append(Variable(w_values*torch.randn(N_neurons, N_neurons), requires_grad=True))
            weights.append(Variable(w_values*torch.randn(label.size, N_neurons), requires_grad=True))

        sig = Sigmoid()
        Ys = []
        Ys.append(sig(weights[0].mm(X)))
        for i in range(N_hidden_layers-1):
            Ys.append(sig(weights[i+1].mm(Ys[-1])))
        Ys.append(weights[N_hidden_layers].mm(Ys[-1]))
        #print("prediction: ", int(Ys[-1].max(0)[1][0]))
        #Y2 = W2.mm(W1.mm(X))
        loss = MSELoss()
        output = loss(Ys[-1], T)
        losses.append(output)
        output.backward()

        for W in weights:
            W.data -= eta* W.grad.data
        for W in weights:
            W.grad.data.zero_()
        
        cpt_blobal +=1
        cpt += 1

        if cpt > Time:
            break

    return [W.data for W in weights]

def prediction(weights, image):
    N_hidden_layers = len(weights)-1
    image = numpy.transpose(numpy.insert(image,0,1))
    image = image.reshape((785, 1))
    X = torch.as_tensor(image)

    sig = Sigmoid()
    Ys = []
    Ys.append(sig(weights[0].mm(X)))
    for i in range(N_hidden_layers-1):
        Ys.append(sig(weights[i+1].mm(Ys[-1])))
    Ys.append(weights[N_hidden_layers].mm(Ys[-1]))
    return int(Ys[-1].max(0)[1][0])


def test_deep(T, weights):
    cpt = 0
    nbOK = 0
    for image,label in test_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        print("value : "+str(numpy.argmax(label)) + " predicted : "+str(prediction(weights, image)))
        if numpy.argmax(label)==prediction(weights, image):
            nbOK += 1
        cpt += 1
        if cpt > T:
            break
    print("OK ratio : "+str(1.0*nbOK/cpt))
    return 1.0*nbOK/cpt

weights = deep(Time=2000, w_values=0.03, eta=1, N_neurons = 32,N_hidden_layers=3)
ratio = test_deep(100, weights)
        
'''etas = []
ratios = []
for i in range(10, 5, -1):
    for j in range(1, 10):
        eta = j*10**(-i)
        print(eta)
        etas.append(eta)
        W1, W2 = perceptron(Time=2000, w_values=0.01, eta=eta)
        ratio = test_perceptron(1000, W)
        ratios.append(ratio)


plt.plot(etas, ratios)
plt.title("Evolution du ratio de bonnes réponses\n en fonction du pas eta")
plt.ylabel("Taux de réussite")
plt.xlabel("Pas d'apprentissage")'''
