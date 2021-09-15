import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist 
#https://projets-lium.univ-lemans.fr/sidekit/_modules/libsvm/svmutil.html
#https://stackoverflow.com/questions/7715138/using-precomputed-kernels-with-libsvm

def precomputed_kernel(x1,x2,gamma):
    #compute kernel matrices between every pairs of (x1,x2) and include sample serial number as first column
    #shape of x1 : (Lx1,784)
    #shape of x2 : (Lx2,784)
    LinearKernel = x1 @ x2.T #=> (Lx1,Lx2)
    RBFKernel = np.exp(-gamma * cdist(x1, x2, 'sqeuclidean')) #=> (Lx1,Lx2); sqeuclidean refers to https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html
    LinearRBFKernel = LinearKernel + RBFKernel #=> (Lx1,Lx2); combine kernel with linear kernel and RBF kernel
    LinearRBFKernel = np.hstack((np.arange(1,len(x1)+1).reshape(-1,1),LinearRBFKernel)) #the training file the first column must be the "ID" of xi. In testing, ? can be any value. refers to https://github.com/cjlin1/libsvm/blob/master/README
    return LinearRBFKernel


if __name__=='__main__':
    #load files and initialize parameters
    trainX = np.genfromtxt('./data/X_train.csv', delimiter=',')
    trainY = np.genfromtxt('./data/Y_train.csv', delimiter=',')
    testX = np.genfromtxt('./data/X_test.csv', delimiter=',')
    testY = np.genfromtxt('./data/Y_test.csv', delimiter=',')
    gamma=0.03125 #is the optimal gamma from HW05_SVM_part2
    
    #defined our own kernels for every pairs of (trainX,trainX) and (testX,trainX)
    kernel_train_train = precomputed_kernel(trainX,trainX,gamma)
    kernel_train_test = precomputed_kernel(testX, trainX, gamma)
    
    #use user-defined kernel to do SVM
    #training
    prob = svm_problem(trainY,kernel_train_train,isKernel=True) #isKernel=True must be set for precomputed kernel. refers to https://github.com/cjlin1/libsvm/blob/master/python/README
    param = svm_parameter('-q -t 4')
    #-q : quiet mode (no outputs)
    #-t kernel_type : set type of kernel function (default 2), kernel_type==4 means precomputed kernel (kernel values in training_set_file)
    model = svm_train(prob,param)
    
    #prediction
    p_label,p_acc,p_vals = svm_predict(testY,kernel_train_test,model,'-q') #"-q" : quiet mode (no outputs).
    print('SVM using linear kernel + RBF kernel together\'s accuracy: {}'.format(p_acc[0]))
    
    #result
    #SVM using linear kernel + RBF kernel together's accuracy: 95.64