  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt	     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def author():  		  	   		     		  		  		    	 		 		   		 		     		     		  		  		    	 		 		   		 		  
    return "Shrikanth Mahale" 		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def seed():  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
    return 903453344  # replace with your GT ID number  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		     		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    result = False  		  	   		     		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		     		  		  		    	 		 		   		 		  
        result = True  		  	   		     		  		  		    	 		 		   		 		  
    return result  		  	 

	  	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    win_prob = 0.474  # set appropriately to the probability of a win  		  	   		     		  		  		    	 		 		   		 		  
    np.random.seed(seed())  # do this only once  		  	   		     		  		  		    	 		 		   		 		  

##########################################################EXPERIMENT 1####################################################################	  	   		     		  		  		    	 		 		   		 		  	          
    def roulette_simple_simulator(total_spins = 1000, episode_winnings_limit = 80):
        spin_number = 1
        episode_winnings = 0
        winnings = np.zeros(total_spins+1,dtype=np.int_)
        while spin_number <= total_spins:
            while episode_winnings < episode_winnings_limit and spin_number <= total_spins:
                bet_amount = 1
                won = False
                while won == False and spin_number <= total_spins:
                    won = get_spin_result(win_prob)
                    if won == True:
                        episode_winnings = episode_winnings + bet_amount
                        winnings[spin_number] = episode_winnings
                    else:
                        episode_winnings = episode_winnings - bet_amount
                        winnings[spin_number] = episode_winnings
                        bet_amount = bet_amount * 2
                    spin_number += 1
            if spin_number <= total_spins:
                winnings[spin_number] = episode_winnings
                spin_number += 1
        return winnings        

    
    #####Experiment 1.1
    plt.clf()
    plt.axis([0, 300, -256, 100])
    plt.axis([0, 300, -256, 100])
    run = 0
    total_run = 10
    while run < total_run:
        winnings = roulette_simple_simulator(total_spins = 1000, episode_winnings_limit = 80)
        run = run+1
        plt.plot(np.arange(0, 1001), winnings)
 #       plt.title("Figure 1: Winnings with 10 Simulated Runs") 
    plt.savefig('Figure_1.png', format='png', dpi=1200)
    
    #####Experiment 1.2
    plt.clf()
    plt.axis([0, 300, -256, 100])
    simulated_winnings = list()
    run = 0
    total_run = 1000
    while run < total_run:
        winnings = roulette_simple_simulator(total_spins = 1000, episode_winnings_limit = 80)
        simulated_winnings.append(winnings)
        run = run+1
        
    Fig2_mean = np.mean(simulated_winnings, axis=0)
    Fig2_stdev = np.std(simulated_winnings, axis=0)
 #   plt.title("Figure 2:Mean Winnings With Unlimited Cash (+/- stdev)")
    plt.plot(Fig2_mean, label = 'mean')
    plt.plot(Fig2_mean + Fig2_stdev , label = 'mean+stdev')
    plt.plot(Fig2_mean - Fig2_stdev, label = 'mean-stdev')
    plt.legend()
    plt.savefig('Figure_2.png', format='png', dpi=1200)

    #####Experiment 1.3
	# Experiment 1 Figure 3
    plt.clf()
    plt.axis([0, 300, -256, 100])
    Fig3_median = np.median(simulated_winnings, axis=0)
    Fig3_stdev  = np.std(simulated_winnings, axis=0)
 #   plt.title("Figure 3: Median Winnings With Unlimited Cash (+/- stdev)")
    plt.plot(Fig3_median, label = 'median')
    plt.plot(Fig3_median + Fig3_stdev, label = 'median+stdev')
    plt.plot(Fig3_median - Fig3_stdev, label = 'median-stdev')
    plt.legend()
    plt.savefig('Figure_3.png', format='png', dpi=1200)
    
##########################################################EXPERIMENT 2####################################################################	
        
    def roulette_realistic_simulator(total_spins = 1000, episode_winnings_limit = 80, betting_limit = 256):
        spin_number = 1
        episode_winnings = 0
        winnings = np.zeros(total_spins+1,dtype=np.int_)
        while spin_number <= total_spins:
            while episode_winnings < episode_winnings_limit and spin_number <= total_spins and episode_winnings != -betting_limit:
                bet_amount = 1
                won = False
                while won == False and spin_number <= total_spins:
                    won = get_spin_result(win_prob)
                    if won == True:
                        episode_winnings = episode_winnings + bet_amount
                        winnings[spin_number] = episode_winnings
                    else:
                        episode_winnings = episode_winnings - bet_amount
                        winnings[spin_number] = episode_winnings
                        if (episode_winnings - (bet_amount*2) > -betting_limit):
                            bet_amount = bet_amount * 2
                        else:
                            bet_amount = episode_winnings + betting_limit
                    spin_number += 1
            if spin_number <= total_spins:
                winnings[spin_number] = episode_winnings
                spin_number += 1
        return winnings
    
    #####Experiment 2.1
    plt.clf()
    plt.axis([0, 300, -256, 100])
    simulated_winnings2 = list()
    run = 0
    total_run = 1000
    while run < total_run:
        winnings = roulette_realistic_simulator(total_spins = 1000, episode_winnings_limit = 80, betting_limit = 256)
        simulated_winnings2.append(winnings)
        run = run+1
        
    Fig4_mean = np.mean(simulated_winnings2, axis=0)
    Fig4_stdev = np.std(simulated_winnings2, axis=0)
 #   plt.title("Figure 4: Mean Winnings With Limited Cash (+/- stdev)")
    plt.plot(Fig4_mean, label = 'mean')
    plt.plot(Fig4_mean + Fig4_stdev , label = 'mean+stdev')
    plt.plot(Fig4_mean - Fig4_stdev, label = 'mean-stdev')
    plt.legend()
    plt.savefig('Figure_4.png', format='png', dpi=1200)
    
    #####Experiment 2.2
    plt.clf()
    plt.axis([0, 300, -256, 100])
    Fig5_median = np.median(simulated_winnings2, axis=0)
    Fig5_stdev = np.std(simulated_winnings2, axis=0)
 #   plt.title("Figure 5: Median Winnings With Limited Cash (+/- stdev)")
    plt.plot(Fig5_median, label = 'median')
    plt.plot(Fig5_median + Fig5_stdev , label = 'median+stdev')
    plt.plot(Fig5_median - Fig5_stdev, label = 'median-stdev')
    plt.legend()
    plt.savefig('Figure_5.png', format='png', dpi=1200)
    
		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
