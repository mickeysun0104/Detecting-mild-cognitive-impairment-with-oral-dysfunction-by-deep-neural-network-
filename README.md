# Detecting mild cognitive impairment with oral dysfunction
Using elders' dental and emotion examination data to predict elders' mild cognitive impairment. The detecting model is able to predict the disease with high sensibility under an extremely imbalanced data distribution. In addition to predicting cognitive impairment, the follow-up research can discover crucial features through visualization of embedding layer and feature engineering of LightGBM. This method provides practical suggestion for the application of clinical diagnosis in the future.

Main skills to use:
- Dealing with imbalanced data with different sampling methods
- A deep learning model to handle high-dimensional features with different data types
- Using LightGBM to do the feature engineering and give proper explanation of the final result
- Visualizing the embedding of training features to do further analysis
![image](https://github.com/mickeysun0104/Detecting-mild-cognitive-impairment-with-oral-dysfunction-by-deep-neural-network-/blob/main/pics/emb_oral6_t6_molar.png)


## Description of the dataset:
  The data in this reasearch is from Health Examination for the Elderly (台北市老人健康檢查資料庫). There are four datasets we used in this reasearch, including oral examination, SPMSQ, BSRS-5 and demographics. The main issues needs to be solved are the sparsity of the oral features and the imbalanced distribution of the target.
![image](https://github.com/mickeysun0104/Detecting-mild-cognitive-impairment-with-oral-dysfunction-by-deep-neural-network-/blob/main/pics/data_sparsity.png)
![image](https://github.com/mickeysun0104/Detecting-mild-cognitive-impairment-with-oral-dysfunction-by-deep-neural-network-/blob/main/pics/spiscore_distribution.png)


## Method
  In order to solve the sparsity of the oral features, a deep neural network is used in the research. [Deep Crossing](https://dl.acm.org/doi/abs/10.1145/2939672.2939704) is the model that we used in the research. It trains embeddings for categorical features to solve high sparsity and dimensionality of data. After training embeddings for categorical features, all features will be the input of Residual Layer to do the further interaction. The structure of the model is shown bellow:
<img src="https://github.com/mickeysun0104/Detecting-mild-cognitive-impairment-with-oral-dysfunction-by-deep-neural-network-/blob/main/pics/deep_crossing.png" width="50%" height="50%"/><br/>
