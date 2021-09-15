import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.spatial.distance import pdist,cdist,squareform
from array2gif import write_gif

#refer to PPT p.22
def DistanceKernel(image_kernel,labels,c):
    len_image_kernel = len(image_kernel)
    k_xj_xj = np.diag(image_kernel).reshape(-1,1) #the similarity between each pixel and itself
    Ck = np.sum(labels == c) #the number of pixels belonging to cluseter c
    k_xj_xn = image_kernel[:, labels == c] #the similarity between each pixel xj and the other pixel xn which belongs to cluster c
    k_xp_xq = image_kernel[labels == c,:][:,labels == c] #the similarity between the pairwise pixels that both belonging to cluster c; first trancate the row, then trancate the column to extract the pairwise pixels both belonging to cluster c
    distance = k_xj_xj - (2/Ck)*np.sum(k_xj_xn,axis=1).reshape(-1,1) + (1/(Ck**2))*np.sum(k_xp_xq) #the distance between each pixel and the center in the feature space
    return distance.reshape(-1)
    

def Kernel(C,gamma_s=1e-3,gamma_c=1e-3):
    height,width = int(np.sqrt(C.shape[0])),int(np.sqrt(C.shape[0]))
    #color information : C =>(10000,3)
    #spatial information : S =>(10000,2)
    S = np.zeros((height*width,2)) #the coordinate of the pixel
    for h in range(height):
        for w in range(width):
            S[height*h+w] = [h,w]
            
    #pdist : compute pairwise distances between observations in "one" n-dimensional space. => 10000*(10000-1)/2
    #cdist : compute distance between each pair of the "two" collections of inputs. => (10000,10000)
    return np.exp(-gamma_s*cdist(S,S,'sqeuclidean')) * np.exp(-gamma_c*cdist(C,C,'sqeuclidean'))

def Initialization(image,k,init_type):
    labels = None #to let first while run
    while len(np.unique(labels))!=k: #until every cluster has at least one pixel
        #initialize k-menas' centers
        height,width,nDimensions = int(np.sqrt(image.shape[0])), int(np.sqrt(image.shape[0])), image.shape[1]
        if init_type == 'kpp':
            #k-means++ ,refer to https://www.cnblogs.com/yixuan-xu/p/6272208.html
            #center initialization
            centers = np.zeros((k,nDimensions)) #initialize k cluseters' center
            centers[0] = image[np.random.randint(height*width)] #randomly choose first cluseter's center
            
            #compute the left k-1 centers
            for c in range(1,k): #k-1 centers c
                #compute the distance of every data points to the nearest center
                for c_computed in range(c): #choose the "nearest" center
                    distance = np.sqrt(np.sum((image - centers[c_computed])**2,axis=1))
                    
                #pick the next center based on the probability of [(distance(X)^2)/(sum of distance(x)^2)], the farer distance is more likly to be chosen
                probability = distance**2 / np.sum(distance**2) #probabiltiy
                mask = np.random.choice(height*width, p=probability) #randomly choose one pixel based on the probability
                centers[c] = image[mask] #assign next center
        
        elif init_type == 'random':
            #random(uniform)
            centers = np.zeros((k,nDimensions)) #initialize k cluseters' center
            mask = np.random.choice(height*width,size=k,replace=False) #randomly(uniform) choose k clusters' center's index without replacement
            for c in range(k):
                centers[c] = image[mask[c]]
        
        #assign each pixel's label to the nearest cluster's center
        distance = np.zeros((k,height*width))
        for c in range(k): #for every centers
            distance[c] = np.sqrt(np.sum((image-centers[c])**2,axis=1))#the distance between each data points and that particular center
        labels = np.argmin(distance,axis=0) #assign the nearest one
    return labels

#refer to https://github.com/algostatml/UNSUPERVISED-ML/blob/master/KMEANS%20AND%20KERNEL%20VERSION/KERNEL%20KMEANS/kernelkmeans.py
def KernelKmeans(image,k,init_type):
    #initialization
    images = [] #to store gif
    labels_diff = 1e+10
    height,width,nDimensions = int(np.sqrt(image.shape[0])), int(np.sqrt(image.shape[0])), image.shape[1]
    colormap = np.array([[255, 255, 255], [255, 255, 0], [255, 0, 255], [255, 0, 0], [0, 255, 255], [0, 255, 0], [0, 0, 255], [0, 0, 0]])#np.random.choice(range(256),size=(k,3)) #randomly pick an color, RGB
    labels0 = Initialization(image,k,init_type) #initialize labels
    
    #set kernel
    image_kernel = Kernel(image)
    
    while labels_diff >= 1e-10:
        distance = np.zeros((k,height*width))
        for c in range(k): #M step: for every centers
            #distance[c] = np.sqrt(np.sum((image_kernel-centers_kernel0[c])**2,axis=1)) #compute the distance between every data points and the centers
            distance[c] = DistanceKernel(image_kernel,labels0,c)
        labels1 = np.argmin(distance,axis=0) #E step: assign the pixel's label with the nearest cluster's center
            
        labels_diff = np.sqrt(np.sum((labels1-labels0)**2))
        labels0 = labels1
        print("difference: ",labels_diff)
        
        #use labels to do visualization
        image_cur = np.zeros((height,width,3))
        for h in range(height):
            for w in range(width):
                image_cur[w,h] = colormap[labels0[height*h+w]] #not [h,w], because labels is from the top to bottom and then from left to right, not from left to right and then from the top to bottom
        images.append(image_cur.astype('uint8')) #store this time's image
        #plt.imshow(image_cur.astype('uint8')) #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        #plt.pause(0.1)
    return images
    

if __name__=='__main__':
    
    #load file and set initialization
    ks = [2,3,4,5]
    init_types = ['kpp','random']
    image_filenames = ['image1','image2']
    
    for image_filename in image_filenames: #for every images
        image = mpimg.imread('./data/{}.png'.format(image_filename)) #=>(100,100,3)
        image = (image * 255).astype('uint8') #convert float32(0~1) to uint8(0~255)
        height,width,colors = image.shape
        image = image.reshape(height*width,colors) #flatten ; =>(10000,3); image[h][w] => image[height*h+w]
        
        #part 2 : in addition to cluster data into 2 clusters, try more clusters (e.g. 3 or 4)
        for k in ks : 
            #part 3 : try different ways to do initialization of k-means clustering eg. k-means++
            for init_type in init_types : 
                #part 1 : show the clustering procedure of kernel k-means
                images=KernelKmeans(image,k,init_type)
                
                #print the process by gif
                write_gif(images, './result/KernelKmeans_{}_{}clusters_{}.gif'.format(image_filename,k,init_type))
                print('KernelKmeans_{}_{}clusters_{} converge!'.format(image_filename,k,init_type))
        
