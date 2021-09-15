import numpy as np
from libsvm.svmutil import * #https://projets-lium.univ-lemans.fr/sidekit/_modules/libsvm/svmutil.html

#load files and initialize parameters
trainX = np.genfromtxt('./data/X_train.csv', delimiter=',')
trainY = np.genfromtxt('./data/Y_train.csv', delimiter=',')
testX = np.genfromtxt('./data/X_test.csv', delimiter=',')
testY = np.genfromtxt('./data/Y_test.csv', delimiter=',')

#grid search
hyperParameters = [2**x for x in range(-5,5)]
accuracyMatrix=np.zeros((len(hyperParameters),len(hyperParameters)))
for IndexC in range(len(hyperParameters)):
    for IndexGamma in range(len(hyperParameters)):
        accuracyMatrix[IndexC,IndexGamma]=svm_train(trainY,trainX,'-q -s 0 -t 2 -v 3 -c {} -g {}'.format(hyperParameters[IndexC],hyperParameters[IndexGamma]))
        #-q : quiet mode (no outputs)
        #-s svm_type : set type of SVM (default 0 => C-SVC)
        #-t kernel_type : set type of kernel function (default 2 => RBF)
        #-v n: n-fold cross validation mode
        #-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        #-g gamma : set gamma in kernel function (default 1/num_features)

#extract the optimal hyperparameters
OptIndexC,OptIndexGamma = np.unravel_index(np.argmax(accuracyMatrix,axis=None),accuracyMatrix.shape)
OptC,OptGamma = hyperParameters[OptIndexC],hyperParameters[OptIndexGamma]
print("accuracy matrix: ")
np.set_printoptions(precision=2, suppress=True)
print(accuracyMatrix)
print("optimal C: {},\noptimal Gamma: {},\noptimal accuracy: {}".format(OptC,OptGamma,accuracyMatrix[OptIndexC,OptIndexGamma]))

#result
#accuracy matrix: 
#[[94.2  73.26 41.86 27.18 21.98 20.88 20.22 59.28 78.86 75.52]
# [96.   74.34 46.4  28.06 22.44 20.54 20.26 39.66 78.84 75.32]
# [96.96 84.54 48.   28.36 21.74 20.68 20.34 46.34 78.94 75.36]
# [97.46 92.54 49.6  34.7  21.68 20.8  20.4  39.66 79.04 75.1 ]
# [97.98 96.82 54.92 44.18 25.14 20.8  20.26 33.12 79.04 75.6 ]
# [98.4  97.86 84.24 62.84 43.62 29.82 24.44 21.9  27.   69.2 ]
# [98.44 97.84 84.9  66.   44.3  32.28 25.16 22.16 20.86 68.94]
# [98.36 97.7  84.98 65.26 45.88 32.08 24.76 21.88 20.78 62.48]
# [98.54 97.9  84.84 65.3  44.72 31.46 25.2  22.12 27.12 62.9 ]
# [98.52 97.94 85.1  65.82 45.36 31.7  24.94 22.14 20.6  69.1 ]]
#optimal C: 8,
#optimal Gamma: 0.03125,
#optimal accuracy: 98.54