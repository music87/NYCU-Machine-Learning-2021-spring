import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import re

def readfiles(dirpath,shape):
    images = []
    labels = []
    #read picture with format of PGM
    for pgm in os.listdir(dirpath): #for each pgm picture
        #deal with individual pgm picture
        filepath = f'{dirpath}/{pgm}'
        image = Image.open(filepath) #use PIL.Image module to read the pgm picture
        image = image.resize(shape, Image.ANTIALIAS) #use PIL.Image module's filter to resize the pgm picture in order to prevent excessive calculations in LDA
        image = np.array(image) #turn the pgm picture to numerical array
        #plt.imshow(image, cmap='gray') #take a look how the picture looks like now
        #plt.show() #you can see the picture even if in debug mode
        image = image.flatten() #flatten the image into 1-D array
        label = int(re.search(r'subject([0-9]+)', pgm).group(1)) #use regular expression to find out the number cancated after the string "subject"
        
        #concate every individual pictures
        images.append(image)
        labels.append(label)
    
    #turn list to array
    images = np.asarray(images,dtype=np.float64) #important!!!!default type=unit8 will lead to kernel computation's overflow!!!!!
    labels = np.asarray(labels)
    return images,labels

def linearKernel(x1,x2):
    return x1 @ x2.T

def rbfKernel(x1, x2, gamma=1e-10): #gamma=1's performance is worse
    return np.exp(-gamma * cdist(x1, x2, 'sqeuclidean'))

def PCA(X,k):
    #build the covariance matrix S, refer to PPT p.119
    meanX = np.mean(X, axis=0)
    covariaceX = (X-meanX) @ (X-meanX).T
    
    #find the orthogonal projection matrix W containing k principal components, refer to PPT p.120
    eigenValues, eigenVectors = np.linalg.eigh(covariaceX)
    if np.all(eigenValues.imag < 1e-3): #if the imaginary part is close to 0
        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real
    kLargestIdx = np.argsort(eigenValues)[::-1][:k] #[::-1] means to revert an array, hence this line code means to find the k "largest" eigenvalues
    W = (X-meanX).T @ eigenVectors[:, kLargestIdx] #project X onto the k principal component's axes
    W = W / np.linalg.norm(W,axis=0) #normalize, because ||w||=1
    
    return W, meanX

def kernelPCA(X,k,kernel): #parameter kerenl is a function
    #implement kernel PCA, refer to PPT p.128
    K = kernel(X,X)
    oneN = np.ones((len(X), len(X))) / len(X)
    Kcov = K - oneN@K - K@oneN + oneN@K@oneN #covariance matrix in the feature space
    eigenValues, eigenVectors = np.linalg.eigh(Kcov)
    if np.all(eigenValues.imag < 1e-3):
        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real
    kLargestIdx = np.argsort(eigenValues)[::-1][:k] #find the k "largest" eigenvalues
    W = eigenVectors[:, kLargestIdx]
    W = W / np.linalg.norm(W,axis=0) #normalize
    return W 


def LDA(X,Y,k):
    #build within-class scatter Sw and between-class scatter Sb ,refer to PPT p.179
    meanX = np.mean(X, axis=0)
    classes = np.unique(Y)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        meanXj = np.mean(X[Y==c],axis=0)
        #within-class scatter Sw
        Sj = (X[Y==c] - meanXj).T @ (X[Y==c] - meanXj)
        Sw += Sj
        
        #between-class scatter Sb
        Sbj = np.sum(Y==c) * ((meanXj-meanX).T @ (meanXj-meanX))
        Sb += Sbj

    #build the projection matrix W, refer to PPT p.181
    eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    if np.all(eigenValues.imag < 1e-3):
        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real
    kLargestIdx = np.argsort(eigenValues)[::-1][:k]
    W = eigenVectors[:, kLargestIdx]
    return W
        
def kernelLDA(X,Y,k,kernel):   
    #build the projection matrix W, refer to formula 15 from [Kernel Eogenfaces vs. kernel fisherfaces: face recognition usgin kernel methods, https://www.csie.ntu.edu.tw/%7Emhyang/papers/fg02.pdf]
    #refer to wiki Multi-class KFD, https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis#Multi-class_KFD
    K = kernel(X,X)
    
    meanK = np.mean(K, axis=0)
    classes = np.unique(Y)
    N = np.zeros((len(X), len(X)))
    M = np.zeros((len(X), len(X)))
    for c in classes:
        
        meanKj = np.mean(K[Y==c], axis=0)
        lj = np.sum(Y==c)
        onelj = np.ones((lj, lj)) / lj
        #within-class scatter N in the feature space
        N += K[Y==c].T @ (np.eye(lj)-onelj) @ K[Y==c]
    
        #between-class scatter M in the feature space
        M += lj * (meanKj-meanK).T @ (meanKj-meanK)
    
    #build the projection matrix W, same logic as PPT p.181
    eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(N) @ M)
    if np.all(eigenValues.imag < 1e-3):
        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real
    kLargestIdx = np.argsort(eigenValues)[::-1][:k]
    W = eigenVectors[:, kLargestIdx]
    return W
    
def showResult(shape, samplesX, W, meanX):
    #show eigenfaces or fisherfaces
    dirpath = "./result"
    os.makedirs(dirpath, exist_ok=True)
    if meanX is None: #LDA
        meanX = np.zeros(samplesX.shape[1])
        alogType = 'LDA'
        faceType = 'Fisherfaces'
    else:  #PCA
        alogType = 'PCA'
        faceType = 'Eigenfaces'
    plt.suptitle(f"{alogType}: {faceType}")
    for i in range(W.shape[1]): #W.shape[1] == 25
        plt.subplot(5,5,i+1)
        plt.imshow(W[:, i].reshape(shape[::-1]), cmap='gray')
        plt.axis('off')
    plt.savefig(f'{dirpath}/part1_{alogType}_{faceType}.png')
    plt.show()
    
    #show original (defalut) 10 samples 
    plt.suptitle(f"{alogType}: Original samples")
    for i in range(len(samplesX)): #len(samplesX) == 10
        plt.subplot(2,5,i+1)
        plt.imshow(samplesX[i].reshape(shape[::-1]),cmap='gray')
        plt.axis('off')
    plt.savefig(f'{dirpath}/part1_{alogType}_original_samples.png')
    plt.show()
    
    #show reconstructed (defalut) 10 samples, refer to PPT p.121
    projectX = (samplesX-meanX) @ W #use projection matrix W to project data in higher dimensional space into lower dimensional space
    reconstructX = projectX @ W.T + meanX #W can project data from high-D into low-D, so in oppsite, W.T can project data from low-D onto high-D
    plt.suptitle(f"{alogType}: Reconstructed samples")
    for i in range(len(reconstructX)): #len(reconstructX) == 10
        plt.subplot(2,5,i+1)
        plt.imshow(reconstructX[i].reshape(shape[::-1]),cmap='gray')
        plt.axis('off')
    plt.savefig(f'{dirpath}/part1_{alogType}_reconstructed_samples.png')
    plt.show()

def KNN(trainX,trainY,testX,acTestY,k): # k nearest neighbor
    distances = cdist(testX,trainX,'sqeuclidean') #compute each train and test's pairwise distance
    predTestY = np.zeros(len(acTestY))
    for i in range(len(testX)): #for each testing data, predict their labels
        kNearestIdx = np.argsort(distances[i])[:k] # find k nearest training data for i-th testing data => compute the k "shortest" distance between i-th testing data and all the training data
        candidates = trainY[kNearestIdx] #possible k predicted labels
        candidates,counts = np.unique(candidates,return_counts=True) #possible unique predicted labels with their counts
        predTestY[i] = candidates[np.argmax(counts)] #major vote
    performance = np.sum(predTestY == acTestY) / len(acTestY)
    return performance

if __name__ == '__main__':
    shape = (65,77)#(195,231) #(width,height)
    #readfiles
    trainX, trainY = readfiles("./Yale_Face_Database/Training",shape)
    testX, testY = readfiles("./Yale_Face_Database/Testing",shape)

    MODE = [1,2,3]
    for mode in MODE:
        if mode == 1:
            print("===================")
            print("Part1:Use PCA and LDA to show the first 25 eigenfaces and fisherfaces, and randomly pick 10 images to show their reconstruction")
            #randomly pick 10 images
            randIdx = np.random.choice(len(trainX), 10) 
            samplesX = trainX[randIdx]
            
            #PCA
            pcaW, meanX = PCA(trainX,25) #find the projection matrix that can project the original(higher) data space to first 25(lower) eigenfaces
            showResult(shape, samplesX, pcaW, meanX) #show the eigenfaces and reconstruction
            
            #LDA
            ldaW = LDA(trainX,trainY,25) #find the projection matrix that can project the original(higher) data space to first 25(lower) fisherfaces
            showResult(shape, samplesX, ldaW, None) #show the eigenfaces and reconstruction
        
        if mode == 2: #reuse the result of part1, so here is not elif
            print("\n===================")
            print("Part2:Use PCA and LDA to do face recognition, and compute the performance. You should use k nearest neighbor to classify which subject the testing image belongs to.")
            #PCA
            ptrainX = (trainX-meanX) @ pcaW #project the training data using PCA that present the 25 pricipal components(from higher dimensional spce into lower dimensional space) of that data
            ptestX = (testX-meanX) @ pcaW #project the testing data using PCA that present the 25 pricipal components(from higher dimensional spce into lower dimensional space) of that data
            performance = KNN(ptrainX,trainY,ptestX,testY,5) #k nearest neighbor, assume k=5
            print(f"KNN's performace under PCA = {performance:>.3f}") #show results
            
            #LDA
            ptrainX = trainX @ ldaW
            ptestX = testX @ ldaW
            performance = KNN(ptrainX,trainY,ptestX,testY,5)
            print(f"KNN's performace under LDA = {performance:>.3f}")
        
        if mode == 3:
            print("\n===================")
            print("Part3:Use kernel PCA and kernel LDA to do face recognition, and compute the performance. (You can choose whatever kernel you want, but you should try different kernels in your implementation.) Then compare the difference between simple LDA/PCA and kernel LDA/PCA, and the difference between different kernels.")
            kernels = [linearKernel,rbfKernel] #function
            for kernel in kernels: #for all kernel functions
                #kernel PCA
                kpcaW = kernelPCA(trainX,25,kernel)
                ptrainX = kernel(trainX,trainX).T @ kpcaW #using kernel projection matrix W, project training data from higher dimensional feature space into lower dimensional feature space, refer to PPT p.128
                ptestX = kernel(trainX,testX).T @ kpcaW #using kernel projection matrix W, project testing data from higher dimensional feature space into lower dimensional feature space, refer to PPT p.128
                performance = KNN(ptrainX,trainY,ptestX,testY,5)
                print(f"KNN's performace under kernel PCA using {kernel.__name__} = {performance:>.3f}")
                
                #kernel LDA
                kldaW = kernelLDA(trainX,trainY,25,kernel)
                ptrainX = kernel(trainX,trainX).T @ kldaW
                ptestX = kernel(trainX,testX).T @ kldaW
                performance = KNN(ptrainX,trainY,ptestX,testY,5)
                print(f"KNN's performace under kernel LDA using {kernel.__name__} = {performance:>.3f}")

    
    
    
    