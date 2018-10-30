# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 08:08:38 2018

@author: Louise
"""

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import math
from torch.autograd import Variable


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

def deep_network(T, w_values, eta, N, taille_N):
    cpt = 0
    cpt_blobal = 0
    for image,label in train_loader:
        
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        print(cpt_blobal, numpy.argmax(label))
        
        dtype = torch.FloatTensor
        if cpt == 0:
            #initialisation
            #matrice de poids 
            W1 = Variable(torch.randn(N,image.size+1).type(dtype), requires_grad=True)
            dW1 = numpy.full(W1.shape, 0.0) 
            
            hidden_weights = [W1]
            hidden_layers = [Variable(torch.randn(N,).type(dtype), requires_grad=False)]
            for n in range(N-1):
                hidden_weights.append(Variable(torch.randn(N,N).type(dtype), requires_grad=True))
                hidden_layers.append(Variable(torch.randn(N,).type(dtype), requires_grad=False))
            
            Ws = Variable(torch.randn(label.size,N).type(dtype), requires_grad=True)
        
        #Propagation de l'activité
        ys = Variable(torch.randn(label.size,).type(dtype), requires_grad=False)
        
        image_tensor = Variable(torch.as_tensor(numpy.transpose(numpy.insert(image,0,1))))
        
        for i, layer in enumerate(hidden_layers):
            if i == 0:
                x = image_tensor
            else:
                x = layer[i-1]
            '''for n in layer:
                
                
                    
                n = 1.0/(1+math.exp(-sum(torch.mm(hidden_weights[i], x))))'''
            tmp = torch.mm(hidden_weights[i], x)
            layer = 1.0/(1+math.exp(-torch.mm(hidden_weights[i], x)))
        
        ys = torch.mm(Ws, hidden_layers[-1])
			
		#Rétro-propagation du gradient
        '''d1 = numpy.full((image.size+1,),0.0)
        d2 = numpy.add(label,-y2)
        for i in range(image.size+1):
            d1[i] = y1[i]*(1-y1[i])*sum(numpy.multiply(d2,W2[:,i]))
			
			
		#Modification des poids
        for i in range(image.size+1):
            for j in range(image.size+1):
                dW1[i,j] = eta*d1[i]*numpy.insert(image,0,1)[j]
		
        for i in range(label.size):
            for j in range(image.size+1):
                dW2[i,j] = eta*d2[i]*y1[j]
        			
        W1 = numpy.add(W1, dW1)
        W2 = numpy.add(W2, dW2)
        cpt_blobal +=1
        cpt += 1
        if cpt > T:
            break'''
    print(ys)
    return ys

def prediction(W1, W2, image):
    y1 = numpy.full((785,),0.0)
    y2 = numpy.full((10,),0.0)
    for i in range(image.size+1):
        y1[i] =  1.0/(1+math.exp(-sum(numpy.multiply(W1[i], numpy.insert(image,0,1)))))
       
    for i in range(W2.shape[0]):
        y2[i] = sum(numpy.multiply(W2[i], y1))
    return numpy.argmax(y2)


def test_perceptron(T, W1, W2):
    cpt = 0
    nbOK = 0
    for image,label in test_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        print("value : "+str(numpy.argmax(label)) + " predicted : "+str(prediction(W1, W2, image)))
        if numpy.argmax(label)==prediction(W1, W2, image):
            nbOK += 1
        cpt += 1
        if cpt > T:
            break
    print("OK ratio : "+str(1.0*nbOK/cpt))
    

ys = deep_network(T=10,N = 10, taille_N = 300, w_values=0.001, eta=0.01)
#test_perceptron(10, W1, W2)
#numpy.max(W)
