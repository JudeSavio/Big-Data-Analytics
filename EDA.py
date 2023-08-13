#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:04:44 2023

@author: judesavio
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import seaborn as sns
from sklearn.decomposition import PCA

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + (nGraphPerRow - 1)) / nGraphPerRow)
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    
    df = df[[col for col in df if df[col].nunique() > 0]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    
    
    corr = df.corr()
    
    # plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    # corrMat = plt.matshow(corr, fignum = 1)
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    # plt.yticks(range(len(corr.columns)), corr.columns)
    # plt.gca().xaxis.tick_bottom()
    # plt.colorbar(corrMat)
    # plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    # plt.show()
    
    ax = sns.heatmap(corr,annot=True)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = None # specify 'None' if want to read whole file

def accessories():
    # accessories.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('accessories.csv', delimiter=',', nrows = nRowsRead)
    df1.dataframeName = 'accessories.csv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns')
    
    print(type(df1['is_new'][0]))
    
    # Coverting is_new column values to discrete
    for i in range(len(df1['is_new'])):
        df1["is_new"][i] = int(df1["is_new"][i])
    
    
    print()
    print('<------------------------------->')
    print('DATASET INFO')
    print()
    print(df1.info())
    print()
    print('<------------------------------->')
    print('DATASET UNIQUE VALUES')
    print()
    print(df1.nunique())
    print()
    print('<------------------------------->')
    print('MISSING VALUE PERCENTAGE')
    print()
    print((df1.isnull().sum()/(len(df1)))*100)
    print()
    print('<------------------------------->')
    print('DATA FRAME DESCRIBE')
    print()
    print(df1.describe())
    print('<------------------------------->')
    print('PCA')
    print()
    X = df1[['current_price' , 'raw_price' ,'discount' , 'likes_count' , 'is_new' , 'id']]
    x_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca_features = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(data=pca_features )
    pca.fit_transform(x_scaled)
    
    
    plt.bar(range(1,len(pca.explained_variance_)+1),pca.explained_variance_)
    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.show()
    print()
    print('PCA Explained Variance : ',pca.explained_variance_ )


    
    # print(df1.describe(include = 'all'))
    
    plotPerColumnDistribution(df1, 10, 5)
    
    plotCorrelationMatrix(df1, 10)
    
    plotScatterMatrix(df1, 15, 10)
    
def jewelry():
    df2 = pd.read_csv('jewelry.csv', delimiter=',', nrows = nRowsRead)
    df2.dataframeName = 'jewelry.csv'
    nRow, nCol = df2.shape
    print(f'There are {nRow} rows and {nCol} columns')
    
    print(df1.info())
    
    plotPerColumnDistribution(df2, 10, 5)
    
    plotCorrelationMatrix(df2, 10)
    
    plotScatterMatrix(df2, 15, 10)

accessories()



