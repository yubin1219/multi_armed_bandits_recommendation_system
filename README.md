# Multi-armed Bandits for Recommendation System
#### This repo is the implementation of reinforcement learning project using Multi-Armed Bandits(MAB) for recommendation systems
## Dataset
#### Anime Recommendation Database 2020 from Kaggle (https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)

## Implemented algorithms
* Epsilon-greedy
* Upper Confidence Bound (UCB)
* Thompson Sampling
## Usage
### Guidance
* Download the anime recommendation dataset and implement `./data preprocessing.py` to filter the top 10 most reviewed anime and obtain rewards and liked percentage per anime.
* `./algorithms.py` is where each MAB algorithm is defined.
* To implement, run this:
```
python3 mab.py
```
* We also provide `anime_recommendation.ipyb` which contains data pre-processing code and main codes for implementing each MAB algorithm.
## Visualize results
* Estimated expected rewards for each algorithm
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/209358937-dde16375-7afd-452e-8327-bf7a40692e47.png" width="80%" height="80%">
</p>
<br/>

* Percentage of recommending optimal animation(e.g. arm4) correctly
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/209358945-92baff8b-2fc7-4a3e-8d6d-cd06bdb729be.png" width="80%" height="80%">
</p>
<br/>

* Percentage of recommending optimal animation(e.g. arm4) during simulation
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/209358943-c3cb9b30-3bfe-4195-82e5-d3bade7f11db.png" width="80%" height="80%">
</p>
<br/>

* Average rewards per iteration
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/209358940-ed31a5f7-968c-40ed-b9b4-91c44225fe25.png" width="80%" height="80%">
</p>
<br/>
