import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress


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



def getTe3(feat_mat):
    TE_df = pd.read_csv('ABIvsTEfeatureProject/PC3聚类版.csv', sep=',', header=0)
    TE_df = TE_df.loc[(TE_df.geneName.isin(feat_mat.row)) , :]
    te = TE_df['TE']
    te_values = te.values.astype(float)
    print(te_values)
    return te_values


def train2(selfeat_mat, feat_mat_col, y, train_size=346, test_size=400):
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


def train(selfeat_mat,feat_mat_col,y):
    featRange = list(range(1, selfeat_mat.shape[1] + 1))  # 特征的范围列表 [1, 2, ..., ncol(selfeat.mat)]
    k = 10  # 将数据集划分为k个折叠
    folds = KFold(n_splits=k, shuffle=True)  # 创建k折交叉验证数据划分器（十折交叉验证划分器）
    performances = []
    for train_ids, test_ids in folds.split(selfeat_mat):
        train_x = selfeat_mat[train_ids][:, np.array(featRange) - 1]  # 取出训练数据特征部分
        train_y = y[train_ids]  # 训练数据的目标变量
        test_x = selfeat_mat[test_ids][:, np.array(featRange) - 1]  # 取出测试数据特征部分
        test_y = y[test_ids]  # 测试数据的目标变量
        train_labels = y[train_ids]
        test_labels = y[test_ids]
        # 绘制训练集和测试集的分布图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(train_labels, bins=10, alpha=0.5, color='blue') * 8
        plt.title('Distribution of Train Labels')
        plt.xlabel('TE')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(test_labels, bins=10, alpha=0.5, color='red')
        plt.title('Distribution of Test Labels')
        plt.xlabel('TE')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


        clf = RandomForestRegressor()  # 创建一个随机森林分类器
        clf.fit(train_x, train_y)  # 训练模型
        pred_y = clf.predict(test_x)  # 预测目标变量
        errors = np.abs(pred_y - test_y)  # 计算每个数据点的误差
        r2 = r2_score(test_y, pred_y)  # 计算R2得分



        plt.plot(test_y, errors, 'o', color='green')
        plt.title('Errors: Actual Value vs Predicted Value')
        plt.xlabel('Actual Value')
        plt.ylabel('errors')
        plt.show()

        # 使用pearsonr函数计算相关性和P值
        correlationR, p_value = pearsonr(pred_y, test_y)
        print("皮尔森相关性系数：", correlationR)
        mean_squared = mean_squared_error(test_y, pred_y)#获取误差方差
        correlation, p_value = spearmanr(pred_y, test_y)#获取相关性和P值

        #画图
        plt.scatter(pred_y, test_y)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        # r_squared = r2_score(pred_y, test_y)
        plt.title('pearsonr Correlation:' + str(correlationR) + '\n' + 'mean_squared:' + str(mean_squared)+ '\n' + 'r2_score:' + str(r2))
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        # plt.plot([-10, 10], [-10, 10], color='red', linestyle='--', label='y = x')
        plt.show()
        print('correlation:' + str(correlation))
        print('mean_squaerd:'+ str(mean_squared))
        print('r2_score:'+ str(r2))
        performances.append(mean_squared)
    print(performances)  # 输出每次交叉验证的性能指标（准确率）

if __name__ == "__main__":
    feat_mat=getfeat_mat()#提取整合数据，化为稀疏矩阵
    selfeat_mat=feat_mat
    print(selfeat_mat.shape)
    te_values = getTe3(feat_mat)#取得TE值
    y = np.log(te_values)#对数化
    selfeat_mat_dense = selfeat_mat.toarray()#将稀疏矩阵化为数组
    print(selfeat_mat_dense)
    train2(selfeat_mat_dense,feat_mat.col,y)#训练，测试数据并评估模型，输出结果


