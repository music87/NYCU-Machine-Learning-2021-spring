import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.spatial.distance import pdist,cdist,squareform
from array2gif import write_gif

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
                #compute the distance of every data points to the nearest: center
                '''
                distance = np.full((height*width),1e+10)
                for s in range(height*width): #for every data points
                    for c_computed in range(c): #choose the "nearest" center
                        distance_cur = np.sqrt(np.sum((image[s] - centers[c_computed])**2))
                        if distance_cur < distance[s] and np.all(image[s]!=centers[c_computed]): #to avoid compare with itself
                            distance[s] = distance_cur        
                '''
                for c_computed in range(c): #choose the "nearest" center
                    distance = np.sqrt(np.sum((image - centers[c_computed])**2,axis=1))
                                        
                #pick the next center based on the probability of [(distance(X)^2)/(sum of distance(x)^2)], the farer D(x) is more likly to be chosen
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
    return centers

#refer to https://github.com/algostatml/UNSUPERVISED-ML/blob/master/KMEANS%20AND%20KERNEL%20VERSION/KERNEL%20KMEANS/kernelkmeans.py
def Kmeans(image,k,init_type):
    #initialization
    images = [] #to store gif
    centers_diff = 1e+10
    height,width,nDimensions = int(np.sqrt(image.shape[0])), int(np.sqrt(image.shape[0])), image.shape[1]
    colormap = np.array([[255, 255, 255], [255, 255, 0], [255, 0, 255], [255, 0, 0], [0, 255, 255], [0, 255, 0], [0, 0, 255], [0, 0, 0]])#np.random.choice(range(256),size=(k,3)) #randomly pick an color, RGB
    centers0 = Initialization(image,k,init_type) #initialize centers
    
    while centers_diff >= 1e-10:
        #E step : classify all samples according to closet centers
        distance = np.zeros((k,height*width))
        for c in range(k): #for every centers
            distance[c] = np.sqrt(np.sum((image-centers0[c])**2,axis=1)) #compute the distance between every data points and the centers
        labels = np.argmin(distance,axis=0) #assign the pixel's label with the nearest cluster's center
        
        #M step : re-compute the centers which are the mean of that cluster
        centers1 = np.zeros((k,nDimensions))
        for c in range(k): #for every centers
            mask = np.argwhere(labels==c).reshape(-1) #find which pixel belongs to cluster c
            centers1[c] = np.average(image[mask],axis=0) #compute new centers
            
        centers_diff = np.sqrt(np.sum((centers1-centers0)**2))
        centers0 = centers1
    
        #use labels to do visualization
        image_cur = np.zeros((height,width,3)) #RGB
        for h in range(height):
            for w in range(width):
                image_cur[w,h] = colormap[labels[height*h+w]] #not [h,w], because labels is from the top to bottom and then from left to right, not from left to right and then from the top to bottom
        images.append(image_cur.astype('uint8')) #store this time's image
        #plt.imshow(image_cur.astype('uint8')) #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        #plt.pause(0.1)
    return images,labels
    

if __name__=='__main__':
    image_filenames = ['image1','image2']
    ks = [2,3,4,5]
    init_types = ['kpp','random']
    
    for image_filename in image_filenames: #for every images
        #load file and set initialization
        image = mpimg.imread('./data/{}.png'.format(image_filename)) #=>(100,100,3)
        image = (image * 255).astype('uint8') #convert float32(0~1) to uint8(0~255)
        height,width,colors = image.shape
        image = image.reshape(height*width,colors) #flatten ; =>(10000,3); image[h][w] => image[height*h+w]
        
        #part 2 : in addition to cluster data into 2 clusters, try more clusters (e.g. 3 or 4)
        for k in ks : #try different number of clusters
            #part 3 : try different ways to do initialization of k-means clustering eg. k-means++
            for init_type in init_types : #try different kmeans' initialized method
                #part 1 : show the clustering procedure of k-means
                images , _ = Kmeans(image,k,init_type)
                
                #print the process by gif
                write_gif(images, './result/Kmeans_{}_{}clusters_{}.gif'.format(image_filename,k,init_type))
                print('Kmeans_{}_{}clusters_{} converge!'.format(image_filename,k,init_type))
        
