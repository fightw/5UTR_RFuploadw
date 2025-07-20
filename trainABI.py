import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


def getfeat_mat():#提取特征信息矩阵
    #读取特征信息的文件
    feat_df = pd.read_table(f"data/input.fa.sparseFeature.txt.gz", compression="gzip")
    # 创建稀疏矩阵
    row = feat_df.iloc[:, 0]  # 第一列作为行索引
    col = feat_df.iloc[:, 1]  # 第二列作为列索引
    data = feat_df.iloc[:, 2]  # 第三列作为非零元素值
    # print(len(data))
    feat_mat = csr_matrix((data, (row, col)))
    #提取序列名
    row_names = pd.read_table(f"data/input.fa.sparseFeature.rowname", header=None)[1]
    #提取特征名
    col_names = pd.read_table(f"data/input.fa.sparseFeature.colname", header=None)[1][:feat_mat.shape[1]]
    # 将 row_names 中的每个元素转换为字符串，并存储为字符串数组
    row_names_array = row_names.astype(str).values
    # 将 row_names 中的每个元素转换为字符串，并存储为字符串数组
    col_names_array = col_names.astype(str).values
    feat_mat.row = row_names_array
    feat_mat.col = col_names_array

    return feat_mat




def getABI(feat_mat):
    ABI_df = pd.read_csv('ABIvsTEfeatureProject/PC3聚类版.csv', sep=',', header=0)
    ABI_df = ABI_df.loc[(ABI_df.geneName.isin(feat_mat.row)) , :]
    te = ABI_df['ABI']
    te_values = te.values.astype(float)
    print(te_values)
    return te_values


from scipy.stats import pearsonr, spearmanr, linregress


def train(selfeat_mat, feat_mat_col, y, train_size=346, test_size=400):
    """
    Train a RandomForestRegressor and evaluate the model performance.

    Parameters:
    - selfeat_mat: Feature matrix for the dataset.
    - feat_mat_col: The number of columns/features in the dataset (not currently used in this function).
    - y: Target variable (dependent variable).
    - train_size: The size of the training dataset (default is 346).
    - test_size: The size of the testing dataset (default is 400).
    """

    # Dynamically create train/test splits based on provided sizes
    featRange = list(range(1, selfeat_mat.shape[1] + 1))
    train_x = selfeat_mat[0:train_size, np.array(featRange) - 1]  # extract training data features
    train_y = y[0:train_size]  # training target variable
    test_x = selfeat_mat[train_size:test_size, np.array(featRange) - 1]  # extract testing data features
    test_y = y[train_size:test_size]  # testing target variable

    # Initialize and train the RandomForestRegressor
    clf = RandomForestRegressor()
    clf.fit(train_x, train_y)

    # Predict the target values
    pred_y = clf.predict(test_x)

    # Calculate various evaluation metrics
    r2 = r2_score(test_y, pred_y)  # calculate R2 score
    mae = mean_absolute_error(test_y, pred_y)  # calculate MAE
    correlationR, p_value = pearsonr(pred_y, test_y)
    mean_squared = mean_squared_error(test_y, pred_y)  # get error variance
    correlation, p_value = spearmanr(pred_y, test_y)  # get correlation and P value

    # Print evaluation metrics
    print("Pearson correlation coefficient:", correlationR)
    print("Mean squared error:", mean_squared)
    print("MAE:", mae)
    print('Spearman correlation coefficient:', correlation)
    print('R2 score:', r2)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_y, test_y, alpha=0.7, color='skyblue', edgecolors='w', s=60)

    # Fit a linear regression line (line of best fit)
    slope, intercept, r_value, p_value, std_err = linregress(pred_y, test_y)
    line_x = np.linspace(-2.5, 2.5, 100)
    line_y = slope * line_x + intercept

    # Plot the regression line with better styling
    plt.plot(line_x, line_y, color='darkorange', linewidth=3, linestyle='-', alpha=0.8)

    # Title and labels
    plt.title(f'Pearson correlation: {correlationR:.2f}\nSpearman correlation: {correlation:.2f}',
              fontsize=18, fontweight='bold', family='Arial')
    plt.xlabel('Predicted Values', fontsize=16, fontweight='bold', family='Arial')
    plt.ylabel('Actual Values', fontsize=16, fontweight='bold', family='Arial')

    # Improve grid and background for aesthetics
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print the results again for clarity
    print('Spearman correlation:', correlation)
    print('Mean squared error:', mean_squared)
    print('R2 score:', r2)

from scipy.sparse import hstack  # 用于将稀疏矩阵进行合并


import shap

if __name__ == "__main__":
    feat_mat=getfeat_mat()#提取整合数据，化为稀疏矩阵
    selfeat_mat=feat_mat
    ABI_values = getABI(feat_mat)#取得TE值
    y = np.log(ABI_values)#对数化
    selfeat_mat_dense = selfeat_mat.toarray()#将稀疏矩阵化为数组
    # train(selfeat_mat_dense,feat_mat.col,y)#训练，测试数据并评估模型，输出结果
    train(selfeat_mat_dense,feat_mat.col,y)#训练，测试数据并评估模型，输出结果



