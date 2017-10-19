# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
d = 10;
N = 10000;
ads_selected = []
# Implementing UCB
# step - 1
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# step - 2
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if(numbers_of_selections[i] > 0):
            #average reward calculation
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            #UCB computation
            upper_bound = average_reward + delta_i
        else:
            #this is so that one of each ad is selected over 'd' rounds
            upper_bound = 1e400
        #calculating max UCB
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            #keep a memory of the ad that has the max upper bound
            ad = i
    #step - 3
    #adding the selected ad to the ads_selected list
    ads_selected.append(ad)
    #incrementing the occurence of that ad in the numbers_of_selections list
    numbers_of_selections[ad] += 1
    #reward from the dataset(GOD KNOWS)
    reward= dataset.values[n,ad]
    #incrementing the reward of that ad in the sum_of_reward list
    sums_of_rewards[ad] += reward
   
#total reward calculation
total_reward = sum(sums_of_rewards)
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()