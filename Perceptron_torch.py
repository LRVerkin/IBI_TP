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

## implementation d'un perceptron simple

def affichage(image,label):
    label = numpy.argmax(label)
    plt.figure()
    plt.imshow(image,interpolation='nearest',cmap='binary')
    plt.title('chiffre '+str(label))
    plt.show()

TRAIN_BATCH_SIZE = 1
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

def perceptron(Time, w_values, eta):
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
            W =  Variable(w_values*torch.ones(label.size, image.size), requires_grad=True) 
        Y = W.mm(X)
        loss = MSELoss()
        output = loss(Y, T)
        output.backward()
        W.data -= eta* W.grad.data
        W.grad.data.zero_()

        cpt_blobal +=1
        cpt += 1
        if cpt > Time:
            break
        
    return W.data

def calcul_activite(poids, inputs):
    Y = poids.data.mm(inputs.data)
    return Y

def prediction(W, image):
    image = numpy.transpose(numpy.insert(image,0,1))
    image = image.reshape((785, 1))
    Y =  W.mm(torch.as_tensor(image))
    return int(Y.max(0)[1][0])


def test_perceptron(T, W):
    cpt = 0
    nbOK = 0
    for image,label in test_loader:
        image = image[0,:].numpy().flatten()
        label = label[0,:].numpy()
        #print("value : "+str(numpy.argmax(label)) + " predicted : "+str(prediction(W, image)))
        if numpy.argmax(label)==prediction(W, image):
            nbOK += 1
        cpt += 1
        if cpt > T:
            break
    print("OK ratio : "+str(1.0*nbOK/cpt))
    return 1.0*nbOK/cpt

W = perceptron(Time=2000, w_values=0.01, eta=0.0005)
ratio = test_perceptron(1000, W)
etas = []
ratios = []
'''for i in range(0, 1, -1):
    for j in range(1, 10):
        eta = j*10**(-i)
        print(eta)
        etas.append(eta)
        W = perceptron(Time=2000, w_values=eta, eta=0.0005)
        ratio = test_perceptron(1000, W)
        ratios.append(ratio)'''
#plot pour le temps
'''for j in range(0, 30):
    eta = j*50
    print(eta)
    etas.append(eta)
    W = perceptron(Time=eta, w_values=0.01, eta=0.0005)
    ratio = test_perceptron(1000, W)
    ratios.append(ratio)

plt.plot(etas, ratios)
plt.title("Evolution du ratio de bonnes réponses\n en fonction du temps d'apprentissage")
plt.ylabel("Taux de réussite")
plt.xlabel("Nombre d'itérations pour l'apprentissage")'''