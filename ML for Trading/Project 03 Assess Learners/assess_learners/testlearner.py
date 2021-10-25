		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		  
import sys  		  	   		     		  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np 
import time
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import LinRegLearner as lrl  		  	   		     		  		  		    	 		 		   		 		  	  	   		    
import util
import matplotlib.pyplot as plt
 		  	   		     		  		  		    	 		 		   		 		 	  	   		     		  		  		    	 		 		   		 		  
def author(self):
    return 'Shrikanth Mahale'
  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  	
    np.seterr(divide='ignore', invalid='ignore')	  	   		     		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		     		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		     		  		  		    	 		 		   		 		  
        sys.exit(1)
    if sys.argv[1] == "Data/Istanbul.csv":
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf,delimiter=',')[1:,1:]
    else:	  	   		     		  		  		    	 		 		   		 		  
        inf = open(sys.argv[1])  		  	   		     		  		  		    	 		 		   		 		  
        data = np.array(  		  	   		     		  		  		    	 		 		   		 		  
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]  		  	   		     		  		  		    	 		 		   		 		  
        )  		
    
    author()		

    ## data = np.genfromtxt(util.get_learner_data_file('Istanbul.csv'), delimiter=',')
    #data = data[1:, 1:]  	
    
	   		     		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		   	  			    		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			    		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # separate out training and testing data  		   	  			    		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]  		   	  			    		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]  

    
    def Decision_Tree_Algorithm(Max_Leaf_Size,learner_class, Verbose,Bags = 20):
        In_Samples_RMSE = np.zeros((Max_Leaf_Size, 1))
        Out_Of_Samples_RMSE = np.zeros((Max_Leaf_Size, 1))
        Time_Frame = In_Samples_RMSE.copy()
        observation = 0
        total_observations = Max_Leaf_Size
        while observation <= total_observations:
            if learner_class == 'DTLearner':
                learner = dt.DTLearner(leaf_size=observation, verbose=Verbose)
            elif learner_class == 'RTLearner':
                learner = rt.RTLearner(leaf_size=observation, verbose=Verbose)
            elif learner_class == 'BagLearner':
                learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":observation}, bags=Bags, boost=False, verbose=Verbose) 
            Start_Time = time.time()
            learner.add_evidence(trainX, trainY)   
            ##In Sample Evaluation
            predY = learner.query(trainX)
            root_mean_squared_error = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            In_Samples_RMSE[observation - 1, 0] = root_mean_squared_error    		  		 	
            ##Out of Sample Evaluation
            predY = learner.query(testX) # get the predictions
            root_mean_squared_error = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            Out_Of_Samples_RMSE[observation - 1, 0] = root_mean_squared_error
            End_Time = time.time()
            Time_Frame[observation - 1, 0] = End_Time - Start_Time
            observation = observation + 1
        return In_Samples_RMSE, Out_Of_Samples_RMSE, Time_Frame, Verbose
        
##########################################################################################################################################################

    #Experiment 1
    Observation = np.arange(1, 101)

    DT_InS_RMSE, DT_OS_RMSE, Time_Frame, DT_Verbose	=    Decision_Tree_Algorithm(100,'DTLearner', True )		 
    plt.plot(Observation, DT_InS_RMSE, label="In Sample RMSE")
    plt.plot(Observation, DT_OS_RMSE, label="Out of Sample RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Square Error")
    plt.legend()
    plt.title("Figure 1: DT Learners Overfitting")
    if DT_Verbose == True:
        plt.savefig("Experiment 1.png") 
    plt.clf()	  
##########################################################################################################################################################

    #Experiment 2
    Observation2 = np.arange(1, 101)

    ## 25 Bag
    Bag_InS_RMSE_1, Bag_OS_RMSE_1, Bag_Time_Frame,  Bag_Verbose_1	=    Decision_Tree_Algorithm(100,'BagLearner', True, 25 )		 
    plt.plot(Observation2, Bag_InS_RMSE_1, label="In Sample RMSE")
    plt.plot(Observation2, Bag_OS_RMSE_1, label="Out of Sample RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Square Error")
    plt.legend()
    plt.title("Figure 2: DT Learners with Bagging Overfitting; Bags = 25    ")
    if Bag_Verbose_1 == True:
        plt.savefig("Experiment 2: Bag - 25.png")
    plt.clf()


##########################################################################################################################################################
    #Experiment 3
    Observation3 = np.arange(1, 101)

    DT_InS_RMSE, DT_OS_RMSE, DT_Time_Frame, DT_Verbose	=    Decision_Tree_Algorithm(100,'DTLearner', True )	
    RT_InS_RMSE, RT_OS_RMSE, RT_Time_Frame, RT_Verbose	=    Decision_Tree_Algorithm(100,'RTLearner', True )	

    plt.plot(Observation3, DT_Time_Frame, label="DT Timeframe")
    plt.plot(Observation3, RT_Time_Frame, label="RT Timeframe")
    plt.xlabel("Leaf Size")
    plt.ylabel("Timeframe")
    plt.legend()
    plt.title("Figure 3: DT Learner Timeframe vs RT Learner Timeframe")
    if DT_Verbose == True | RT_Verbose == True :
        plt.savefig("Experiment 3.png")
    plt.clf()


##########################################################################################################################################################