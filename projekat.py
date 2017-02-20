# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:02:07 2016

@author: Luka
"""

from sklearn import datasets
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from scipy.misc import imresize
from skimage.color import rgb2gray
from skimage.morphology import square, diamond, disk
from skimage.morphology import opening, closing, dilation
from skimage.measure import label  # implementacija connected-components labelling postupka
from skimage.measure import regionprops 
import collections
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift, estimate_bandwidth
from math import sqrt

brojevi=np.zeros([10045,784],'uint8')



niz=[]
r=0
for x in range(1,11):
    for y in range(1,1000):
        slika=imread("Sample"+str('{0:03}'.format(x))+"/img"+str('{0:03}'.format(x))+"-00"+str('{0:03}'.format(y))+".png")
        binar=1-(slika>30)
        binar=dilation(binar)
        labeled=label(binar)
        regions=regionprops(labeled)
        for region in regions:
            bbox=region.bbox          
            img_crop =  slika[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            geg=imresize(img_crop,[28,28])
            z=geg.reshape(784)
            brojevi[(x-1)*999+(y-1)]=z
        #niz.append(z)
print "pocearo"

for x in range (1,11):
    proba=imread("plus"+str(x)+".png")
    gray=rgb2gray(proba)*255
    binar=1-(gray>30)
    binar=dilation(binar)
    labeled=label(binar)
    regions=regionprops(labeled)
    for region in regions:
        bbox=region.bbox          
        img_crop =  gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        geg=imresize(img_crop,[28,28])
        z=geg.reshape(784)
        brojevi[9990+x-1]=z
    
for x in range (1,11):
    proba=imread("minus"+str(x)+".png")
    gray=rgb2gray(proba)*255
    binar=1-(gray>30)
    binar=dilation(binar)
    labeled=label(binar)
    regions=regionprops(labeled)
    for region in regions:
        bbox=region.bbox          
        img_crop =  gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        geg=imresize(img_crop,[28,28])
        z=geg.reshape(784)
        brojevi[10000+x-1]=z

for x in range (1,21):
    proba=imread("puta"+str(x)+".png")
    gray=rgb2gray(proba)*255
    binar=1-(gray>30)
    binar=dilation(binar)
    labeled=label(binar)
    regions=regionprops(labeled)
    for region in regions:
        bbox=region.bbox          
        img_crop =  gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        geg=imresize(img_crop,[28,28])
        z=geg.reshape(784)
        brojevi[10010+x-1]=z

for x in range (1,11):
    proba=imread("podijeljeno"+str(x)+".png")
    gray=rgb2gray(proba)*255
    binar=1-(gray>30)
    binar=dilation(binar)
    labeled=label(binar)
    regions=regionprops(labeled)
    for region in regions:
        bbox=region.bbox          
        img_crop =  gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        geg=imresize(img_crop,[28,28])
        z=geg.reshape(784)
        brojevi[10030+x-1]=z


for x in range (1,6):
    proba=imread("korijen"+str(x)+".png")
    gray=rgb2gray(proba)*255
    binar=1-(gray>30)
    binar=dilation(binar)
    labeled=label(binar)
    regions=regionprops(labeled)
    for region in regions:
        bbox=region.bbox          
        img_crop =  gray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        geg=imresize(img_crop,[28,28])
        z=geg.reshape(784)
        brojevi[10040+x-1]=z

#print niz.shape
nbrs=NearestNeighbors(1,'auto').fit(brojevi)
#filename = 'finalized_model.sav'
#pickle.dump(nbrs, open(filename, 'wb'))
print "zavrsio"





'''
proba=imread("Sample001/img001-00025.png")
proba=imresize(proba,[28,28])
proba=proba.reshape(1,784)


dobijeno=nbrs.kneighbors(proba)
'''
#daj1=dobijeno[1].reshape(28,28)

#plt.imshow(daj1,'gray')


proba=imread("problem1.png")

gray=rgb2gray(proba)*255

binar=1-(gray>30)   
binar=dilation(binar)
plt.imshow(binar,'gray')
labeled=label(binar)
regions=regionprops(labeled)
grayu=gray.astype('uint8')


zz=[]
tt=[]
regioni = {}
regions.sort
for region in regions:
    bbox=region.bbox  
    #plt.imshow(grayu[bbox[0]-10:bbox[2]+10,bbox[1]-10:bbox[3]+10],'gray')
    img_crop =  grayu[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    proba=imresize(img_crop,[28,28])
    proba=proba.reshape(1,784)
    dobijeno=nbrs.kneighbors(proba)
    zz.append(dobijeno[1][0][0])
    regioni[bbox]=dobijeno[1][0][0]
    

def sortByKey():
    sortirani=sorted(regioni.items(), key = lambda t : t[0][1])
    return sortirani
   
    
centers = [[1, 1], [-1, -1], [1, -1]]

sortirani=sortByKey()

sortiraniNiz=np.asarray(sortirani)
sortNiz=sortiraniNiz[:,0]



centri=np.zeros([sortNiz.shape[0],1])

for x,y in enumerate(sortNiz):
    centri[x]=[y[0]+(y[2]-y[0])/2]


#print sortirani
             
#print iris.data
#print centri
#print sortirani
dbscan=DBSCAN(eps=25, metric='euclidean',min_samples=2,algorithm='ball_tree').fit(centri)

labels=dbscan.labels_

unique_labels = set(labels)
print unique_labels



matrica=[]


         
for index1,x in enumerate(unique_labels):
    new=[]
    for index,l in enumerate(labels):
        if x==l:         
            new.append(sortiraniNiz[index])
    matrica.append(new)

for i,z in enumerate(unique_labels):
    izraz=matrica[i]
    odnos=0
    kraj_korijena=0
    sekvenca=''
    #collections.OrderedDict(sorted(regioni.items()))
    for x, broj in izraz:
        if(odnos==0):
            odnos=x[2]-x[0]
        if((x[2]-x[0])-odnos<0 and broj<9990):
            sekvenca+="**"
        if(kraj_korijena!=0 and kraj_korijena<x[1]):
            sekvenca+=')'
            kraj_korijena=0
        if broj < 999:
            sekvenca+=str(0)
        elif broj < 1999:
            sekvenca+=str(1)
        elif broj < 2997:
            sekvenca+=str(2)
        elif broj < 3996:
            sekvenca+=str(3)
        elif broj < 4995:
            sekvenca+=str(4)
        elif broj < 5994:
            sekvenca+=str(5)
        elif broj < 6993:
            sekvenca+=str(6)
        elif broj < 7992:
            sekvenca+=str(7)
        elif broj < 8991:
            sekvenca+=str(8)
        elif broj < 9990:
            sekvenca+=str(9)
        elif broj < 10000:
            sekvenca+='+'
        elif broj < 10010:
            sekvenca+='-'
        elif broj < 10030:
            sekvenca+='*'
        elif broj < 10040:
            sekvenca+='/'
        elif broj < 10045:
            sekvenca+='sqrt('
            kraj_korijena=x[3]
            if(odnos==x[2]-x[0]):
                odnos=0
            
            
    if(kraj_korijena!=0):
        sekvenca+=')'        
    print sekvenca
    print eval(sekvenca)
#plt.imshow(slikica,'gray')
#plt.imshow(kraj,'gray')
