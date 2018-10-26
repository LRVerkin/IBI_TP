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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# 10 fois
#for i in range(0,10):
    # on demande les prochaines données de la base
#    (_,(image,label)) = enumerate(train_loader).next()
    # on les affiche
#    affichage(image[0,:].numpy(),label[0,:].numpy())
# NB pour lire (plus proprement) toute la base (ce que vous devrez faire dans le TP) plutôt utiliser la formulation suivante

def perceptron(T, w_values, eta):
    cpt = 0
    cpt_blobal = 0
    for image,label in train_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        print(cpt_blobal, numpy.argmax(label))
        if cpt == 0:
            #initialisation
            #matrice de poids 
            W = numpy.full((label.size,image.size+1),w_values)
            dW = numpy.full(W.shape, 0.0) 
        
        y = numpy.full((10,),0.0)
        for i in range(label.size):
            y[i] =  sum(numpy.multiply(W[i], numpy.insert(image,0,1)))
            for j in range(image.size+1):
                dW[i,j] = eta*numpy.insert(image,0,1)[j]*(label[i]-y[i])
        print(y)
        W = numpy.add(W, dW)
        cpt_blobal +=1
        cpt += 1
        if cpt > T:
            break
    return W

def prediction(W, image):
    y = numpy.full((10,),0)
    for i in range(W.shape[0]):
        y[i] =  sum(numpy.multiply(W[i], numpy.insert(image,0,1)))
    return numpy.argmax(y)


def test_perceptron(T, W):
    cpt = 0
    nbOK = 0
    for image,label in test_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        print("value : "+str(numpy.argmax(label)) + " predicted : "+str(prediction(W, image)))
        if numpy.argmax(label)==prediction(W, image):
            nbOK += 1
        cpt += 1
        if cpt > T:
            break
    print("OK ratio : "+str(1.0*nbOK/cpt))
    

W = perceptron(T=1000, w_values=0.001, eta=0.0001)
test_perceptron(100, W)
numpy.max(W)
