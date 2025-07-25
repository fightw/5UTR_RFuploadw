# Machine Learning-Based Analysis of the Impact of 5'  Untranslated Region on Protein Expression

The purpose of this project is to predict mRNA translation efficiency and protein expression levels, as well as model visualisation, through the following steps: 
 1. Feature extraction
 2. Model training and evaluation
 3. Model visualisation

## 1.Feature extraction

Feature extraction includes the following：
1.mRNA minimum free energy (MFE)
2.k-mer frequencies (k=1-6)
3.uAUG counts upstream of main start codons
4.GC content metrics
5.G-quadruplex ΔMFE values
6.base-pairing distances

### Running the code:
```
python FeatureExtraction_xlsx.py
```


## 2.Model training and evaluation

The random forest regression model was implemented in Python 3.8.18 using scikit-learn (v1.3.0), comprising 100 decision trees (n_estimators=100) with key hyperparameters including max_features=0.8 and min_samples_leaf=1.
To control the randomness of each decision tree, the ExtraTreeRegressor was employed (14). The Final predictions were derived from averaging the outputs of all decision trees. This ensemble learning approach effectively leverages the independence of multiple trees, reducing the risk of overfitting while enhancing the model's ability to fit complex data patterns

### Running the code:
#### TE prediction
```
python trainModel.py
```

#### ABI prediction
```
python trainABI.py
```

## 3.Model visualisation
This module uses the Feature_importance of the random forest model to display the native importance of each feature in the model, as well as SHAP plots to analyse the contribution of each feature to the model output, and difference heatmaps to find feature preferences for sequences with high and low TE values.

#   Dependencies and versions
Here we list the specific versions of individual dependencies needed:
|Software|Version|
|--|--|
| ViennaRNA |2.6.2  |
|python  | 3.11.4 |
| multiprocessing | 0.6.0 |
| sklearn| 1.3.0|
| conda| 23.7.4|
| biopython| 1.81|
