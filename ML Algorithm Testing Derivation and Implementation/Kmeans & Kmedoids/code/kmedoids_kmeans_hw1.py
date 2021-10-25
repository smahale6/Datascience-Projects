from scipy.spatial import distance
import numpy as np
import copy

def fn_kmeans(input_image, K):
    centerpoint = input_image[np.random.randint(low = 0, high =input_image.shape[0], size=K),:].astype(float)
    bkp_K = copy.deepcopy(K)
    group_images = np.zeros(input_image.shape[0]).astype(int)

    def cal_centers(group_images, K, centerpoint): 
        for m in range(K):
            if (group_images==m).any(): 
                upd_centerpoint = np.mean(input_image[group_images==m],axis=0)
                centerpoint[m] = upd_centerpoint
            else: 
                centerpoint[m] = np.nan 
                K = K-1 
        centerpoint = centerpoint[~np.isnan(centerpoint).any(axis=1)] 
        return centerpoint, K
    
    def cal_distance (group_images, centerpoint, K): 
        edist = distance.cdist(input_image,centerpoint,'euclidean') 
        for m in range(len(group_images)): 
            group_images[m] = edist[m].argmin()
        centerpoint, K = cal_centers(group_images, K, centerpoint) 
        return group_images, centerpoint, K
    n =0
    while n<=200: 
        org_centerpoint = centerpoint.copy() 
        group_images, new_centerpoint, K = cal_distance(group_images, centerpoint, K)
        if new_centerpoint.shape == org_centerpoint.shape:
            ct = new_centerpoint == org_centerpoint 
            if ct.all() == False:
                centerpoint = new_centerpoint
            else:
                centerpoint = new_centerpoint
                break
        else:
            centerpoint = new_centerpoint
        n += 1
    print ("Number of k-means iteration:", n)
    if K !=bkp_K:
        print ("Starting with K=", bkp_K, "Empty cluster - K too large. Deleting empty cluster,\n new K=",K)
    return group_images, centerpoint

def fn_kmedoids(input_image, K):
    centerpoint = input_image[np.random.randint(low = 0, high =input_image.shape[0], size=K),:]
    group_images = np.zeros(input_image.shape[0]).astype(int)
    def cal_centers(input_image, centerpoint, group_images):
        for m in range(K):
            upd_centerpoint = np.mean(input_image[group_images==m],axis=0)
            centerpoint[m] = upd_centerpoint 
        dist = distance.cdist(input_image, centerpoint, 'cityblock')
        for m in range(K):
            index = dist[:,m].argmin() 
            centerpoint[m] = input_image[index] 
        return centerpoint
    def cal_distance(group_images, centerpoint):
        mdist = distance.cdist(input_image,centerpoint,'cityblock') 
        for m in range(len(group_images)):
            group_images[m] = mdist[m].argmin()
        centerpoint = cal_centers(input_image,centerpoint, group_images)
        return group_images, centerpoint
    n =0
    while n<=200:
        org_centerpoint = centerpoint.copy()
        group_images, new_centerpoint = cal_distance(group_images, centerpoint)
        ct = new_centerpoint == org_centerpoint
        if ct.all() == False:
            centerpoint = new_centerpoint
        else:
             centerpoint = new_centerpoint
             break
        n += 1
    print ("Number of k-medoids iteration:", n)
    return group_images, centerpoint