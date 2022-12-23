import numpy as np

def normalize(array):
    maximum=max(array)
    minimum=min(array)
    for i in range(len(array)):
        array[i]=(array[i]-minimum)/(maximum-minimum)
    return array


def epsilon_greedy(k,eps,valid_recommendations,anime_arm_id,logged_data):
    
    # k : number of arms/number of total anime to be recommended
    # eps : exploration fraction-> algorithm will explore with a probability of eps and exploit with a probability of (1-eps)
    # valid_recommendations : total number of recommendations to be simulated in the online evaluation
    # anime_arm_id : dictionary mapping arm number to anime id 
    # logged_data : data log to simulate online evaluation through the offline replayer method

    if len(anime_arm_id)!=k: 
        print("The length of list of anime id's passed does not match the number of arms entered \n")
        return
    
    q = np.zeros(k)                   # the array of the estimated expected values of each arm/anime
    recommendations = 0               # 총 추천 횟수
    anime_recom_number = np.zeros(k)  # 각 애니메이션이 추천된 횟수 저장
    reward = 0                        # 현재의 추천으로 얻은 보상
    a = 0                             # 현재 추천하는 애니메이션
    total_reward = 0                  # 총 보상
    avg_reward = np.zeros(valid_recommendations) # 현재까지의 추천 당 평균 보상
    
    while(recommendations != valid_recommendations): # 설정한 추천 횟수에 도달할 때까지 진행
        
        # drawing sample of size k times more than logged_data with replacement to implement bootstrapping
        sample = logged_data.sample(frac = k, replace=True)
        
        for i in range(len(sample)):
            
            # select arm
            p = np.random.rand() # randomly generates a number between 0 and 1
        
            if eps == 0 and recommendations == 0: # eps value indicated exploitation but since the reccomendations made are also zero there is no knowledge to exploit-randomly select any arm
                a = np.random.choice(k) # anime is chosen
        
            elif p > eps: # case of exploration
                a = np.argmax(q) # recommends the anime with the highest estimated expected value at current moment of time
            
            else: # case of exploitation
                a = np.random.choice(k) # anime is chosen randomly
            
            # checking if this recommendation is valid 
            
            if(sample.iloc[i][0]==anime_arm_id[a]): # valid point->consider this for evaluation
                
                # storing the corresponding reward
                reward = sample.iloc[i][1]
                
                # updating counts
                recommendations+=1
                anime_recom_number[a]+=1
                
                # updating the rewards
                total_reward+=reward
                avg_reward[recommendations-1] = total_reward/recommendations
                
                # updating estimated expected value of the recommended anime
                q[a]=((q[a]*(anime_recom_number[a]-1))+reward)/anime_recom_number[a]
                
                if(recommendations==valid_recommendations): # stops the evaluation if valid number of recommendations have been made
                    break
                
        
    avg_reward = normalize(avg_reward) #fits the average rewards between a scale of 0 and 1
    best = np.argmax(q)
    
    # returns the estimated expected values of all thr anime after all the recommendations
    # the average normalised reward per iteration at each iteration
    # the number of recommendations made for each anime
    # the best anime arm number
    
    return best, q, avg_reward, anime_recom_number 

def ucb_1(k,valid_recommendations,anime_arm_id,logged_data):

    if len(anime_arm_id)!=k: 
        print("The length of list of anime id's passed does not match the number of arms entered \n")
        return
    
    q = np.zeros(k)                   # the array of the estimated expected values of each arm/anime
    recommendations = 0               # 총 추천 횟수
    anime_recom_number=np.zeros(k)    # 각 애니메이션이 추천된 횟수 저장
    reward=0                          # 현재의 추천으로 얻은 보상
    a=0                               # 현재 추천하는 애니메이션
    total_reward = 0                  # 총 보상
    avg_reward=np.zeros(valid_recommendations) # the average reward per recommendation till the current recommendation
    round = 0

    # first each anime has to be recommended once
    for a in range(k): # for each anime
        round+=1
        # filtering data of "anime a" from logged data
        filt=(logged_data["anime_id"]==anime_arm_id[a]) 
        a_data=logged_data[filt]
        
        # choosing a random row in the filtered data
        i=np.random.choice(len(a_data))
        reward=a_data.iloc[i][1] # storing the reward for the particular anime

        # updating counts
        recommendations+=1
        anime_recom_number[a]+=1
                
        # updating the rewards
        total_reward+=reward
        avg_reward[recommendations-1]=total_reward/recommendations
                
        # updating estimated expected value of the recommended anime
        q[a]=((q[a]*(anime_recom_number[a]-1))+reward)/anime_recom_number[a]
        
    
    # now each anime has been recommended once, continuing with the algorithm
    while(recommendations!=valid_recommendations):
        
        # drawing sample of size k times more than logged_data with replacement to implement bootstrapping
        sample=logged_data.sample(frac=k,replace=True)
        
        for i in range(len(sample)):
            round+=1
            # select arm 
            upper_bound_arm = q + np.sqrt(np.divide(2*np.log10(round),anime_recom_number))
            a=np.argmax(upper_bound_arm) # choosing the anime with maximum upper bound
            
            # checking if this recommendation is valid 
            if(sample.iloc[i][0]==anime_arm_id[a]): 
                
                # storing the corresponding reward
                reward=sample.iloc[i][1]
                
                # updating counts
                recommendations+=1
                anime_recom_number[a]+=1
                
                # updating the rewards
                total_reward+=reward
                avg_reward[recommendations-1]=total_reward/recommendations
                
                # updating estimated expected value of the recommended anime
                q[a]=((q[a]*(anime_recom_number[a]-1))+reward)/anime_recom_number[a]
                
                if(recommendations==valid_recommendations): # stops the evaluation if valid number of recommendations have been made
                    break
                
        
    avg_reward=normalize(avg_reward) # fits the average rewards between a scale of 0 and 1
    best=np.argmax(q)

    return best,q,avg_reward,anime_recom_number 

def thompson_sampling(k,valid_recommendations,anime_arm_id,logged_data):

    if len(anime_arm_id)!=k: 
        print("The length of list of anime id's passed does not match the number of arms entered \n")
        return
    
    q = np.zeros(k)                   # the array of the estimated expected values of each arm/anime
    recommendations = 0               # 총 추천 횟수
    anime_recom_number=np.zeros(k)     # 각 애니메이션이 추천된 횟수 저장
    reward=0                          # 현재의 추천으로 얻은 보상
    a=0                               # 현재 추천하는 애니메이션
    total_reward = 0                  # 총 보상
    avg_reward = np.zeros(valid_recommendations) # the average reward per recommendation till the current recommendation
    
    # a and b for each of anime has been initialised to 1
    # this will correspond to each arm having a uniform distribution initially
    a_item = np.ones(k)
    b_item = np.ones(k)
    
    
    while(recommendations!=valid_recommendations):
        
        # drawing sample of size k times more than logged_data with replacement to implement bootstrapping
        sample=logged_data.sample(frac=k,replace=True)
        
        for i in range(len(sample)):
            
            # select arm
            # choose the arm which maximises the value returned from the beta function
            beta_val_item=np.ones(k) # holds the value from the beta distribution for each item
            # sample a value from the beta distribution of all k arms
            for j in range(k):
                beta_val_item[j] = np.random.beta(a_item[j],b_item[j])
            # pull the arm whose sampled value is high
            a = np.argmax(beta_val_item)
            
            # checking if this recommendation is valid 
            if(sample.iloc[i][0]==anime_arm_id[a]):
                
                # storing the corresponding reward
                reward = sample.iloc[i][1]
                
                # update alpha or beta value
                if(reward==1): # success
                    a_item[a]+=1
                else:          # failure
                    b_item[a]+=1
                
                # updating counts
                recommendations+=1
                anime_recom_number[a]+=1
                
                # updating rewards
                total_reward+=reward
                avg_reward[recommendations-1]=total_reward/recommendations
                
                # updating estimated expected value of the recommended item
                q[a]=((q[a]*(anime_recom_number[a]-1))+reward)/anime_recom_number[a]
                
                if (recommendations==valid_recommendations): # stops the evaluation if valid number of recommendations have been made
                    break
                
        
    avg_reward_normalize = normalize(avg_reward)
    best=np.argmax(q)

    return best, q, avg_reward_normalize, anime_recom_number