#!pip install libsvm #https://projets-lium.univ-lemans.fr/sidekit/_modules/libsvm/svmutil.html
from libsvm.svmutil import *
import numpy as np

#load files
trainX = np.genfromtxt('./data/X_train.csv', delimiter=',')
trainY = np.genfromtxt('./data/Y_train.csv', delimiter=',')
testX = np.genfromtxt('./data/X_test.csv', delimiter=',')
testY = np.genfromtxt('./data/Y_test.csv', delimiter=',')

#use different kernels and see their performance
kernel = ['linear','polynomial','RBF']
for kernel_type in range(3): #-t kernel_type, kernel_type=0/1/2 means linear/polynomial/RBF kernel respectively
    model = svm_train(trainY,trainX,'-q -t {}'.format(kernel_type)) #-q means quiet mode (no outputs)
    p_label,p_acc,p_vals = svm_predict(testY,testX,model,'-q')
    #p_labels means a list of predicted labels
    #p_acc: a tuple including  accuracy (for classification), mean-squared error, and squared correlation coefficient (for regression).
    #p_vals: a list of decision values or probability estimates
               #(if \'-b 1\' is specified). If k is the number of classes,
               #for decision values, each element includes results of predicting k(k-1)/2 binary-class SVMs.
               #For probabilities, each element contains k values indicating the probability that the testing instance
               #is in each class.
    print('{} kernel\'s accuracy: {:.2f}'.format(kernel[kernel_type],p_acc[0]))

#result
#linear kernel's accuracy: 95.08
#polynomial kernel's accuracy: 34.68
#RBF kernel's accuracy: 95.32