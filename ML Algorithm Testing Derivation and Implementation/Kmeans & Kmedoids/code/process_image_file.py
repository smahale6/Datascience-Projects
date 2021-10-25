import matplotlib.pyplot as mat_plot
import numpy as np
from kmedoids_kmeans_hw1 import fn_kmeans
from kmedoids_kmeans_hw1 import fn_kmedoids

def process_image(img_name, K):
    image = mat_plot.imread(img_name)
    row = image.shape[0]
    column = image.shape[1]
    pixel = np.zeros((row*column,3))
    for m in range(row):
        for n in range(column):
            pixel[n*row+m,:] = image[m,n,:]
    ctr1, ctrd1 = fn_kmeans(pixel, K)
    ctr2, ctrd2 = fn_kmedoids(pixel, K)
    n_image_1 = np.zeros((row, column, 3))
    n_image_2 = np.zeros((row, column, 3))
    for m in range(row):
        for n in range(column):
            n_image_1[m,n,:] = ctrd1[ctr1[n*row+m],:]
            n_image_2[m,n,:] = ctrd2[ctr2[n*row+m],:]    
    n_image_1 /= 255
    n_image_2 /= 255
    mat_plot.subplot(2,2,1)
    mat_plot.title('Input Image')
    mat_plot.imshow(image)
    mat_plot.axis('off')
    mat_plot.subplot(2,2,3)
    mat_plot.title('k-means output')
    mat_plot.imshow(n_image_1)
    mat_plot.axis('off')
    mat_plot.subplot(2,2,4)
    mat_plot.title('k-medoids output')
    mat_plot.imshow(n_image_2)
    mat_plot.axis('off')
    mat_plot.show()
    return None

process_image('butterfly.bmp',32)
