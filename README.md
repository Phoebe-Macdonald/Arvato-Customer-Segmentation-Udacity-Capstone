# Arvato-Customer-Segmentation-Udacity-Capstone

A full explanation of the project motivation, methodology and results can be found here: https://medium.com/@phoebe.macdonald/who-are-arvato-banks-most-valuable-customers-7504f6bf3c77
## Motivation:
With the rise of disruptive fin-techs and general public uncertainty, it is more important than ever for banks to understand who are their customers and how to best to communicate with them.
This project aims to solve this for Arvato Financial Solutions by applying both unsupervised and supervised machine learning techniques.
This project had who aims:
- Identify who Arvato customers are and the qualities that distinguish them from the general population
- Understand how Arvato bank can optimise efficacy and efficiency of marketing communication with their customers, with a particular focus on a direct mail-out campaign

## Summary of results

To achieve the first objective, a k-modes clustering algorithm was refined and developed.
Results were visualised post t-SNE reduction and evaluated using the Silhouette method
Distributions of the customer dataset and the general population were compared across clusters. 
The general population were described by one cluster. There were 5 additional clusters within the customers dataset.
Individuals in these cluster differed from the general population in terms of their age and wealth - generally they were older, higher earners and of a higher class.

To achieve the second objective, three deep-learning algorithms were trained, evaluated and compared. 
The best model achieved and AUC under the ROC curve of 0.57 and a AUC under the Precision Recall curve of 0.02.
These figures indicate that the model was better than a random classifier at identifying customers likely to respond to a direct mail-out.
The model was then  used to make predictions on a new dataset and results uploaded to a Kaggle competition: https://www.kaggle.com/c/udacity-arvato-identify-customers


## Prerequisites:
Data preparation:
- import numpy as np
- import pandas as pd
- import re

Data visualisations:
- import matplotlib.pyplot as plt
- import seaborn as sns
- from matplotlib import pyplot
- from mpl_toolkits.mplot3d import Axes3D

Data processing:
- from sklearn import preprocessing
- from sklearn.preprocessing import StandardScaler
- from sklearn.decomposition import PCA
- from sklearn.manifold import TSNE

Customer segmentation:
- from sklearn.cluster import KMeans
- from kmodes.kmodes import KModes
- from sklearn.metrics import silhouette_score
- from sklearn.metrics import silhouette_samples

Predictive models:
- from catboost import CatBoostClassifier
- import lightgbm as lgb
- import xgboost as xgb

Model evaluation: 
- from sklearn.model_selection import train_test_split
- from sklearn.model_selection import GridSearchCV
- from sklearn import metrics
- from sklearn.metrics import roc_curve
- from sklearn.metrics import precision_recall_curve

## Credits:
My fantastic mentor Marom and manager Jamie and the following links:
- Visualising distribution of data: https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c
- Visualising correlations between metrics https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c 
- Label Encoding: https://stackoverflow.com/questions/42196589/any-way-to-get-mappings-of-a-label-encoder-in-python-pandas and https://www.kaggle.com/ashydv/bank-customer-clustering-k-modes-clustering
- Establishing optimum number of components in PCA https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe 
- Establishing optimum number of clusters for kmeans https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3
- Applying Kmodes: https://www.kaggle.com/ashydv/bank-customer-clustering-k-modes-clustering
- Visualising clusters in 2D using t-SNE https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
- Applying Silhouette method https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- Applying LightGBM, Catboost and XGBoost algorithms: https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
- Plotting ROC curve of model https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/ 
- Guidance on how to create and tune an XGBoost model https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/



## Files:
- For data protection reasons, the data used in this project is not available for download from this GitHub repository.
- The full analysis and workflow can be found in: Arvato Project Workbook.ipynb/html 
- A summary of results can be found on the following Medium article: https://medium.com/@phoebe.macdonald/who-are-arvato-banks-most-valuable-customers-7504f6bf3c77


