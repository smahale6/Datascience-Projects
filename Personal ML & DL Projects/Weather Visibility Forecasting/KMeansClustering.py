import pandas as pd
import numpy as np
from preprocessing import data_preprocessing
import matplotlib.pyplot as plt
from pathlib import Path
import os
from activity_logger import activity_logger
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.model_selection import train_test_split


class KMeansClustering:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def elbow_diagram(self,file_object,X,Image_Folder): 
        self.activity_log_obj.log(file_object, 'Entered the elbow diagram method of the KMeansClustering class')
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++') # initializing the KMeans object
                kmeans.fit(X) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss,marker='o') # creating the graph between WCSS and the number of clusters
            plt.title(Image_Folder +'The Elbow Diagram')
            plt.xlabel('Number of clusters')
            x_ticks = np.arange(0, 10, 1)
            plt.xticks(x_ticks)
            plt.ylabel('SSE')
            plt.savefig(Image_Folder +'K-Means_Elbow.PNG') # saving the elbow plot locally
            plt.show()
            plt.clf()
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.activity_log_obj.log(file_object, 'The optimum number of clusters is: '+str(kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return kn.knee

        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            self.activity_log_obj.log(file_object,'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()
        
    def cluster_creation(self,file_object,X,y,Image_Folder):
        try:    
            knee = self.elbow_diagram(file_object,X,Image_Folder)
            self.activity_log_obj.log(file_object, 'Adding cluster number to the training dataframe.')
            km = KMeans(n_clusters=knee)
            Cluster_Number = km.fit_predict(X)
            X['Cluster'] = Cluster_Number
            self.activity_log_obj.log(file_object, 'Added cluster number to the training dataframe.')
            if len(X) == len(y):
                return X,y,km
            else:
                raise ValueError("X and y have different number of records, hence cannot be used for model.")
                self.activity_log_obj.log(file_object, 'X and y have different number of records, hence cannot be used for model.')
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.activity_log_obj.log(file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
        
if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath("training_validation_insertion"))
    data = pd.read_csv(file_path +'\\Data_Training\\input.csv')

    Image_Folder = file_path +'\\Images\\' 
    Path(Image_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Image_Folder))

    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    data_preprocessing_obj = data_preprocessing()
    KMeansClustering_obj = KMeansClustering()
    
    final_data = data_preprocessing_obj.feature_engineering(file_object,data,Image_Folder)
    X,y = data_preprocessing_obj.feature_scaling_splitting(file_object,final_data,'VISIBILITY','NoScale')
    X,y,km = KMeansClustering_obj.cluster_creation(file_object,X,y,Image_Folder)
    
    file_object.close()
    
    
    
    
