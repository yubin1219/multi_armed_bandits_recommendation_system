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
* To implement, run this:
```
python3 mab.py
```
* We also provide `anime_recommendation.ipyb` which contains data pre-processing code and main codes for implementing each MAB algorithm.
## Results
