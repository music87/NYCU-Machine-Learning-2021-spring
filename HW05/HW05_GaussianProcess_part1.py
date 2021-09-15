import numpy as np
from HW05_GaussianProcess_util import *
if __name__=='__main__':
    #load file and initialize parameters
    trainX,trainY=load_file()
    beta=5
    
    #prediction refer to PPT p.48
    C = kernel(trainX,trainX) + 1/beta * np.identity(len(trainX)) #we can compute the similarity between random variables s by covariance because the covariance is actually the kernel, refer to PPT p.45    
    testX=np.linspace(-60,60,num=100) #create test data in [-60,60]
    testY_mean = kernel(trainX,testX).T @ np.linalg.inv(C) @ trainY.reshape(-1,1) #use gaussian regression to predict the mean of test y
    testY_mean = testY_mean.reshape(-1) #for plot readibly
    testY_variance = kernel(testX,testX)+1/beta*np.identity(len(testX)) - kernel(trainX,testX).T @ np.linalg.inv(C) @ kernel(trainX,testX) #use gaussian regression to predict the variance of test y
    testY_sd = np.sqrt(np.diag(testY_variance)) #only need variance(x1,x1), not variance(x1,x2), and we convert it from variance to standard deviation
    
    #visualization
    drawing(trainX,trainY,testX,testY_mean,testY_sd)
