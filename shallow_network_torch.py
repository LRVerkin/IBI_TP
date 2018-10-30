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

def shallow(Time, w_values, eta, N_neurons):
    cpt = 0
    cpt_blobal = 0
    for image,label in train_loader:
        image = numpy.transpose(numpy.insert(image[0,:].numpy().flatten(), 0, 1))
        image = image.reshape((785,1))
        label = numpy.transpose(label[0,:].numpy())
        label = label.reshape((10,1))
        #print(cpt_blobal, numpy.argmax(label))
        
        T = Variable(torch.as_tensor(label), requires_grad = False)
        X = Variable(torch.as_tensor(image), requires_grad = True)
        
        if cpt == 0:
            #initialisation
            #matrice de poids 
            #losses = []
            W1 =  Variable(w_values*torch.randn(N_neurons, image.size), requires_grad=True) 
            W2 =  Variable(w_values*torch.randn(label.size, N_neurons), requires_grad=True) 
        
        #propagation de l'activité
        sig = Sigmoid()
        Y1 = sig(W1.mm(X))
        Y2 = W2.mm(Y1)
        
        #retropropagation de l'erreur
        loss = MSELoss()
        output = loss(Y2, T)
        #losses.append(output)
        output.backward()
        W1.data -= eta* W1.grad.data
        W2.data -= eta* W2.grad.data
        
        W1.grad.data.zero_()
        W2.grad.data.zero_()
        
        cpt_blobal +=1
        cpt += 1
        if cpt > Time:
            break
        #plt.plot(losses)
    return W1.data, W2.data

def prediction(W1, W2, image):
    image = numpy.transpose(numpy.insert(image,0,1))
    image = image.reshape((785, 1))
    X = torch.as_tensor(image)
    Y1 = 1/(1+torch.exp(-W1.mm(X)))
    Y2 = W2.mm(Y1)
    return int(Y2.max(0)[1][0])


def test_shallow(T, W1, W2):
    cpt = 0
    nbOK = 0
    for image,label in test_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        #print("value : "+str(numpy.argmax(label)) + " predicted : "+str(prediction(W1, W2, image)))
        if numpy.argmax(label)==prediction(W1, W2, image):
            nbOK += 1
        cpt += 1
        if cpt > T:
            break
    print("OK ratio : "+str(1.0*nbOK/cpt))
    return 1.0*nbOK/cpt

#W1, W2 = shallow(Time=3000, w_values=0.03, eta=0.06, N_neurons = 32)
#ratio = test_shallow(100, W1, W2)

etas = []
ratios = []
results = numpy.zeros((20, 20))
ns = [50,100,150,200,250,300,350,400,450, 500,550,600,650,700,750]
for i in range(0, 20):
    for j in range(1,12):
        eta = i/20
        #w = j/20
        print("eta : ", eta," n : ", j)
        etas.append(eta)
        W1, W2 = shallow(Time=1500, w_values=eta, eta=0.06, N_neurons = j*50)
        ratio = test_shallow(500, W1, W2)
        results[i,j] = ratio
        #ratios.append(ratio)

plt.imshow(results)
#0.01, 0.06, 32
#plt.plot(etas, ratios)
plt.title("Evolution du ratio de bonnes réponses\n en fonction des valeurs de poids initiales")
#plt.ylabel("Taux de réussite")
#plt.xlabel("omega")