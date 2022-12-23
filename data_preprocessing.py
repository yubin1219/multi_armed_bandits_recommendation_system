import pandas as pd
import numpy as np

# load the data
df=pd.read_csv("rating_complete.csv")

# 가장 많이 평가된 anime top10
top_10=np.array(df['anime_id'].value_counts().sort_values(ascending=False).head(10).index)

# generate filtered data
filtered_rating=df[df['anime_id'].isin(top_10)]

# load the information for anime
anime_df=pd.read_csv('anime.csv')

# top 10 anime에 대한 데이터만 가져오기
anime_names = anime_df[anime_df['MAL_ID'].isin(top_10)]

anime_dic=dict(zip(anime_names["MAL_ID"],anime_names["Name"]))

filtered_rating['title']=filtered_rating['anime_id'].apply(lambda x: anime_dic[x])

# obtain rewards
filtered_rating['reward']=filtered_rating['rating'].apply(lambda x: 0 if x<10 else 1)

# calculating the most liked anime out of the top 10 most reviewed movies
groups=filtered_rating.groupby("title")

anime_title=[]
anime_liked_percentage=[]
for title, title_df in groups:
    anime_title.append(title)
    anime_liked_percentage.append((np.sum(title_df["reward"])/len(title_df))*100)

liked_per_dic=dict(zip(anime_title,anime_liked_percentage))

# save filtered data
filtered_rating.to_csv('final_anime.csv')

# obtain liked percentage per anime
liked_per=pd.DataFrame.from_dict(liked_per_dic,orient="index")
liked_per.to_csv("liked_per.csv")

anime_id_name=pd.DataFrame.from_dict(anime_dic,orient="index")
anime_id_name.to_csv("anime_id_name.csv")