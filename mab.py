import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import savetxt
from algorithms import epsilon_greedy, ucb_1, thompson_sampling

# load dataset
dataset = pd.read_csv("final_anime.csv")

# mapping anime_id to anime title
anime_id_name=pd.read_csv("anime_id_name.csv")
anime_id_name.columns=["anime_id","anime_title"]

anime_id=anime_id_name["anime_id"]
anime_title=anime_id_name["anime_title"]
anime_id_name=dict(zip(anime_id,anime_title))

# mapping anime title to anime liked percentage
liked_per=pd.read_csv("liked_per.csv")
liked_per.columns=["anime_title","liked_percentage"]

anime_title=liked_per["anime_title"]
anime_liked_per=liked_per["liked_percentage"]
liked_per=dict(zip(anime_title,anime_liked_per))

# defining logged data- contains only sequences of anime id/rewards
# 2 columns- anime_id, reward
logged_data=dataset.drop(labels=["Unnamed: 0","user_id","rating","title"],axis=1,inplace=False)

# dictionary mapping arm number to anime id
anime_arm_id=dict(zip([i for i in range(10)],anime_id))

# Implement MAB
k=10
eps=0.1
valid_recommendations=1000
episodes=1000 #each epsisode is an experiment of all algorithms with 1000 valid recommendations

#long term q
q_eps_long=np.zeros(k)
q_ucb_long=np.zeros(k)
q_ts_long=np.zeros(k)

#long term avg rewards
avg_reward_eps_long=np.zeros(valid_recommendations)
avg_reward_ucb_long=np.zeros(valid_recommendations)
avg_reward_ts_long=np.zeros(valid_recommendations)

#long term anime recom number
anime_recom_number_eps_long=np.zeros(k)
anime_recom_number_ucb_long=np.zeros(k)
anime_recom_number_ts_long=np.zeros(k)


#best arm->number of times each arm has been returned as the best arm
best_arm_eps=np.zeros(k)
best_arm_ucb=np.zeros(k)
best_arm_ts=np.zeros(k)


for i in tqdm(range(episodes)):
    
    best_eps,q_eps,avg_reward_eps,anime_recom_number_eps=epsilon_greedy(k,eps,valid_recommendations,anime_arm_id,logged_data)
    best_ucb,q_ucb,avg_reward_ucb,anime_recom_number_ucb=ucb_1(k,valid_recommendations,anime_arm_id,logged_data)
    best_ts,q_ts,avg_reward_ts,anime_recom_number_ts=thompson_sampling(k,valid_recommendations,anime_arm_id,logged_data)
    
    #long term estimated expected rewards
    q_eps_long=q_eps_long+(q_eps-q_eps_long)/(i+1)
    q_ucb_long=q_ucb_long+(q_ucb-q_ucb_long)/(i+1)
    q_ts_long=q_ts_long+(q_ts-q_ts_long)/(i+1)
    
    #long term avg rewards
    avg_reward_eps_long=avg_reward_eps_long+(avg_reward_eps-avg_reward_eps_long)/(i+1)
    avg_reward_ucb_long=avg_reward_ucb_long+(avg_reward_ucb-avg_reward_ucb_long)/(i+1)
    avg_reward_ts_long=avg_reward_ts_long+(avg_reward_ts-avg_reward_ts_long)/(i+1)
    
    
    #long term anime recom number
    anime_recom_number_eps_long=anime_recom_number_eps_long+(anime_recom_number_eps-anime_recom_number_eps_long)/(i+1)
    anime_recom_number_ucb_long=anime_recom_number_ucb_long+(anime_recom_number_ucb-anime_recom_number_ucb_long)/(i+1)
    anime_recom_number_ts_long=anime_recom_number_ts_long+(anime_recom_number_ts-anime_recom_number_ts_long)/(i+1)
    
    #best arm updates
    best_arm_eps[best_eps]+=1
    best_arm_ucb[best_ucb]+=1
    best_arm_ts[best_ts]+=1


## Results ##
# save the results
# long term q
savetxt('q_eps_long.csv', q_eps_long, delimiter=',')
savetxt('q_ucb_long.csv', q_ucb_long, delimiter=',')
savetxt('q_ts_long.csv', q_ts_long, delimiter=',')

#long term avg rewards
savetxt('avg_reward_eps_long.csv', avg_reward_eps_long, delimiter=',')
savetxt('avg_reward_ucb_long.csv', avg_reward_ucb_long, delimiter=',')
savetxt('avg_reward_ts_long.csv', avg_reward_ts_long, delimiter=',')

#long term anime recom number
savetxt('anime_recom_number_eps_long.csv', anime_recom_number_eps_long, delimiter=',')
savetxt('anime_recom_number_ucb_long.csv', anime_recom_number_ucb_long, delimiter=',')
savetxt('anime_recom_number_ts_long.csv', anime_recom_number_ts_long, delimiter=',')

#best arm->number of times each arm has been returned as the best arm
savetxt('best_arm_eps.csv', best_arm_eps, delimiter=',')
savetxt('best_arm_ucb.csv', best_arm_ucb, delimiter=',')
savetxt('best_arm_ts.csv', best_arm_ts, delimiter=',')

true_values=[liked_per[anime_id_name[i]]/100 for i in anime_id_name.keys()]
true_values

expected_values=np.vstack((true_values,q_eps_long,
                          q_ucb_long,q_ts_long,))
data=pd.DataFrame(expected_values)

data.columns=anime_id_name.values()
data.index=["True Expected Rewards based on liked percentage",
            "Estimated Expected Reward for epsilon=0.1",
            "Estimated Expected Reward for UCB1",
            "Estimated Expected Reward for Thompson Sampling"
]

data.to_csv("estimated_expected_rewards.csv")