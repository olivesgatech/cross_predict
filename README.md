# Learning Trajectory-conditioned Relations to Predict Pedestrian Crossing Behavior

---
This work was mainly done in the [Omni Lab for Intelligent Visual Engineering and Science (OLIVES)](https://ghassanalregib.info/) @ Georgia Tech.
Feel free to check our lab's [Website](https://ghassanalregib.info/) and [GitHub](https://github.com/olivesgatech) for other interesting work!!!
[<img align="right" src="https://www.dropbox.com/s/rowej0iof65fie5/OLIVES-logo_with_website.png?raw=1" width="15%">](https://ghassanalregib.info/)

---
C. Zhou, G. AlRegib, A. Parchami, and K. Singh, "Learning Trajectory-Conditioned Relations to Predict Pedestrian Crossing Behavior," in *IEEE International Conference on Image Processing (ICIP)*, Bordeaux, France, Oct. 16-19 2022.



### Abstract
In smart transportation, intelligent systems avoid potential collisions by predicting the intent of traffic agents, espe- cially pedestrians. Pedestrian intent, defined as future action, e.g., start crossing, can be dependent on traffic surround- ings. In this paper, we develop a framework to incorporate such dependency given observed pedestrian trajectory and scene frames. Our framework first encodes regional joint information between a pedestrian and surroundings over time into feature-map vectors. The global relation representations are then extracted from pairwise feature-map vectors to esti- mate intent with past trajectory condition. We evaluate our approach on two public datasets and compare against two state-of-the-art approaches. The experimental results demon- strate that our method helps to inform potential risks during crossing events with 0.04 improvement in F1-score on JAAD dataset and 0.01 improvement in recall on PIE dataset. Fur- thermore, we conduct ablation experiments to confirm the contribution of the relation extraction in our framework.

---
### Usage
Training/validation: Run the following command in a bash script.
```
python3 core/main_intent.py --config_file configs_intent/shallowCNN/rn_PIE.yml --gpu 0 
```
