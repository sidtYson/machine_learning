# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
d = 10;
N = 10000;
ads_selected = []
# Implementing Thompson sampling
# step - 1
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
# step - 2
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        #step - 2
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        #calculating max UCB
        if random_beta > max_random :
            max_random = random_beta
            #keep a memory of the ad that has the max random beta
            ad = i
    #step - 3
    #adding the selected ad to the ads_selected list
    ads_selected.append(ad)
    #reward from the dataset(GOD KNOWS)
    reward= dataset.values[n,ad]
    #updating the numbers_of_rewards_1 and numbers_of_rewards_0 based on
    # the reward
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    #total reward
    total_reward += reward
        
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()