#
#  sne.py
#
# Implementation of t-SNE and symmetric-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython sne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import imageio
import io
import os
from scipy.spatial.distance import cdist

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def sne(algo, X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = sne.sne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    Ys = [] #modify original tsne to my tsne
    Ps = [] #modify, same as above
    Qs = [] #modify, same as above
    iters = [] #modify, same as above

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        #sum_Y = np.sum(np.square(Y), 1)
        
        #modify original tsne to my tsne and ssne
        #Part1: Try to modify the code a little bit and make it back to symmetric SNE
        if algo == 'tSNE': #refer to PPT p.172
            #num = -2. * np.dot(Y, Y.T) 
            #num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num = 1 / ( 1 + cdist(Y,Y,'sqeuclidean') ) #cdist(Y, Y, 'sqeuclidean') means ||Y-Y||^norm2
        elif algo == 'sSNE': #refer to PPT p.167
            num = np.exp( -1 * cdist(Y,Y,'sqeuclidean') )
        num[range(n), range(n)] = 0. #diagonal
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            #modify original tsne to my tsne and ssne
            if algo == 'tSNE': #refer to PPT p.172
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            elif algo == 'sSNE': #refer to PPT p.167
                dY[i, :] = np.dot( PQ[i,:] , Y[i,:]-Y ) #sum up all j(hidden in :)
            

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            Ys.append(Y) #modify original tsne to my tsne ; to record the the convergence process and do the visualization
            Ps.append(P) #modify, same as above
            Qs.append(Q) #modify, same as above
            iters.append(str(iter)) #modify, same as above

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    # Return solution
    return Ys,Ps,Qs,iters #modify original tsne to my tsne

def visualization(algo, Ys, Ps, Qs, iters, perplexity=30.0): #modify original tsne to my tsne
    #part 2 : Visualize the embedding of both t-SNE and symmetric SNE. Details of the visualization
    #Project all your data onto 2D space and mark the data points into different colors respectively. The color of the data points depends on the label.
    #Use videos or GIF images to show the optimize procedure.
    dirpath = f'./result/part2_{algo}_2Dprojection_{perplexity}perplexity'
    bufferMode = True
    if not bufferMode: os.makedirs(dirpath, exist_ok=True)
    images = []
    for i,Y in zip(iters,Ys):
        pylab.title(f"{algo} 2D projection with {perplexity} perplexity")
        pylab.scatter(Y[:, 0], Y[:, 1], s=20, c=labels) #s means the marker size, c means the marker's color;
        if bufferMode: 
            buffer = io.BytesIO()
            pylab.savefig(buffer, format='png') 
            buffer.seek(0)
            pylab.show()
            image = pylab.imread(buffer, format='png')
            #buffer.truncate(0) #buffer.tell() #buffer.getvalue() #buffer.close()
            buffer.close()
        else:
            pylab.savefig(f'{dirpath}/iteration{i}.png', format='png')
            pylab.show()
            image = pylab.imread(f'{dirpath}/iteration{i}.png', format='png')
        images.append(image)
    imageio.mimsave(f'./result/part2_{algo}_2Dprojection_{perplexity}perplexity.gif', images) #save as gif
    
    #part 3 : Visualize the distribution of pairwise similarities in both high-dimensional space and low-dimensional space, based on both t-SNE and symmetric SNE
    Dimensionalities = ['high','low']
    Data = [Ps[-1],Qs[-1]]
    for dim,data in zip(Dimensionalities,Data):
        pylab.suptitle(f"in {dim} dimensional space with {perplexity} perplexity")
        pylab.subplot(1,2,1)
        pylab.title('the distribution of similarity') #the log distribution of pairwise similarities in high-D
        pylab.xlabel("pairwise similarity")
        pylab.ylabel("log probability")
        pylab.hist(data.flatten(),bins=100,log=True)
        pylab.subplot(1,2,2)
        pylab.title('the similarity heat matrix') #the pairwise similarities heat matrix in high-D
        pylab.xlabel("data points")
        pylab.ylabel("data points")
        pylab.imshow(data, cmap='binary', interpolation='nearest')
        pylab.savefig(f'./result/part2_{algo}_similarity_{dim}D_{perplexity}perplexity.png', format='png')
        pylab.show()


if __name__ == "__main__":
    print("Run Y = sne.sne(X, no_dims, perplexity) to perform t-SNE and symmetric-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("./SNE_Database/mnist2500_X.txt") #modify original tsne to my tsne; X has 784 dimensions
    labels = np.loadtxt("./SNE_Database/mnist2500_labels.txt") #modify original tsne to my tsne; Y has 2 dimensions
    perplexities = [10.0, 20.0, 30.0, 40.0, 50.0, 100, 300, 500, 700, 900] #refer to PPT p.163
    algos = ['tSNE','sSNE'] #sSNE means symmetric SNE
    for algo in algos:
        #Part4:Try to play with different perplexity values. Observe the change in visualization and explain it in the report
        for perplexity in perplexities:
            Ys,Ps,Qs,iters = sne(algo, X, 2, 50, perplexity) #modify original tsne to my tsne
            visualization(algo, Ys,Ps,Qs,iters,perplexity) #modify original tsne to my tsne
