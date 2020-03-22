# Arvato-Customer-Segmentation-Udacity-Capstone

## Motivation:
Bank Arvato want to increase understanding of their customers in order to improve efficiency of their mail out processes.
In order do this, this project:
- developed a segmentation model to understand how the company's customer base differed from the general population
- developed a predictive model to identify customers who are most likely to respond to a mailout
- evaluated the effectiveness of the predictive model

Results were uploaded to Kaggle to compare model accuracy vs similar attempts: https://www.kaggle.com/c/udacity-arvato-identify-customers
A blog post was created to summarise the results: https://medium.com/@phoebe.macdonald/who-are-arvato-banks-most-valuable-customers-7504f6bf3c77

## Prerequisites:
Data preparation:
- import numpy as np
- import pandas as pd
- import re

Data visualisations:
- import matplotlib.pyplot as plt
- import seaborn as sns
- from matplotlib import pyplot

Customer segmentation:
- from sklearn.preprocessing import StandardScaler
- from sklearn.decomposition import PCA
- from sklearn.cluster import KMeans
- from sklearn import preprocessing

Predictive model:
- from sklearn.model_selection import train_test_split
- import xgboost as xgb
- from sklearn.metrics import roc_curve
- from sklearn.metrics import roc_auc_score

## Credits:
- Visualising distribution of data: https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c
- Visualising correlations between metrics https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c 
- Establishing optimum number of components in PCA https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe 
- Establishing optimum number of clusters for kmeans https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3
- Plotting ROC curve of model https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/ 
- Guidance on how to create and tune an XGBoost model https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/



## Files:
For data protection reasons, the data used in this project is not available for download from this GitHub repository.
The full analysis and workflow can be found in: Arvato Project Workbook.ipynb/html 
A summary of results can be found on the following Medium article: https://medium.com/@phoebe.macdonald/who-are-arvato-banks-most-valuable-customers-7504f6bf3c77


