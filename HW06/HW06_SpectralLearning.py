import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
from scipy.spatial.distance import pdist,cdist,squareform
from array2gif import write_gif
from HW06_KernelKmeans import Kernel #self-defined
from HW06_Kmeans import Kmeans #self-defined

def plot_eigen(k, U, labels,result_filepath):
    colormap = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'gray']
    for c in range(k): #for every cluster
        #actually you can choose any other two dimension to see how it looks like for different angle
        plt.scatter(U[labels == c, 0], #the first dimension of coordinate of data points in c cluster
                    U[labels == c, 1], #the second dimension of coordinate of data points in c cluster
                    c=colormap[c], s=1)
    plt.savefig(result_filepath)
    plt.show()

def compute_eigen(init_cut_type,image_filename,k,L):
    
    file_path_eigenvalue = './HW06_SpectralLearning_{}_{}Cut_eigenvalues.npy'.format(image_filename,init_cut_type)
    file_path_eigenvector = './HW06_SpectralLearning_{}_{}Cut_eigenvectors.npy'.format(image_filename,init_cut_type)
    
    if os.path.isfile(file_path_eigenvalue): # if the pre-computed eigen file exist
        eigenvalues = np.load(file_path_eigenvalue)
        eigenvectors = np.load(file_path_eigenvector) 
    else:
        eigenvalues, eigenvectors = np.linalg.eig(L) #compute the eigens, slow; the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]; the eigenvalues are not necessarily ordered
        np.save(file_path_eigenvalue,eigenvalues) #store it to avoid compute the eigenvalues slowly again
        np.save(file_path_eigenvector,eigenvectors) #store it to avoid compute the eigenvectors slowly again
    
    #eigen decomposition may return a complex number with small imaginary part, so here turns it to the real number; refer to https://stackoverflow.com/questions/60366008/numpy-always-gets-complex-eigenvalues-and-wrong-eigenvectors
    if np.all(eigenvalues.imag < 1e-10): #if the imaginary part is close to 0
        eigenvalues = eigenvalues.real #np.real_if_close(eigenvalues)
        eigenvectors = eigenvectors.real #np.real_if_close(eigenvectors)
    
    k_smallest_nonzero_eigenvalues_index = np.argsort(eigenvalues)[1:1+k] #k smallest "non-zero" eigenvalue
    U = eigenvectors[:,k_smallest_nonzero_eigenvalues_index] #refer to PPT p.53
    return U

if __name__=='__main__':
    
    image_filenames = ['image1','image2']
    k = 5
    init_kmeans_type = 'kpp'
    init_cut_types = ['unnormalized','normalized'] #relaxing Ncut leads to normalized spectral clustering, while relaxing RatioCut leads to unnormalized spectral clustering , refer to A Tutorial on spectral clustering, https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf
    
    for image_filename in image_filenames: #for every images
        #load file and set initialization
        image = mpimg.imread('./data/{}.png'.format(image_filename)) #=>(100,100,3)
        image = (image * 255).astype('uint8') #convert float32(0~1) to uint8(0~255)
        height,width,colors = image.shape
        image = image.reshape(height*width,colors) #flatten ; =>(10000,3); image[h][w] => image[height*h+w]
        W = Kernel(image) #similarity matrix
        D = np.diag(np.sum(W,axis=1)) #degree matrix, refer to PPT p.28
        L = D - W #graph Laplacian, refer to PPT p.38
        
        #part 3 : try different ways to do initialization of spectral clustering eg. ratio cut, normalized cut
        for init_cut_type in init_cut_types:
            #======unnormalized(ratio) cut version, refer to PPT p.56,66======#
            if init_cut_type == 'unnormalized':
                U = compute_eigen(init_cut_type,image_filename,k,L) #U contains the first k eigenvectors of L, refer to PPT p.65,66 (H there is U here)
                images , labels = Kmeans(U,k,init_kmeans_type)
            
            #======normalized cut version, refer to PPT p.73======#
            elif init_cut_type == 'normalized':
                Lsym = np.diag(np.diag(D)**(-1/2)) @ L @ np.diag(np.diag(D)**(-1/2)) #normalized Laplacian Lsym = D^{-1/2}LD^{-1/2} = I-D^{-1/2}WD^{-1/2} refer to PPT p.71,72
                U = compute_eigen(init_cut_type,image_filename,k,Lsym) #U contains the first k eigenvectors of Lsym, refer to PPT p.70,73
                T = U / np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1) # refer to PPT p.73
                images , labels = Kmeans(T,k,init_kmeans_type)
        
            #part 1 : show the clustering procedure of spectral clustering
            result_filepath = './result/SpectralLearning_{}_{}_{}clusters_{}'.format(image_filename,init_cut_type,k,init_kmeans_type)
            write_gif(images, '{}.gif'.format(result_filepath)) #print the process by gif
            print('{} converge!'.format(result_filepath))
            
            #part 4: examine whether the data points within the same cluster do have the same coordinates in the eigenspace of graph Laplacian or not
            plot_eigen(k, U, labels, '{}.png'.format(result_filepath))
    
