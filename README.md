# Machine_learning_project-supervised-learning

## Project/Goals

The goal of the project is to perform a full supervised learning machine learning project on a "Diabetes" dataset. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. [Kaggle Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

## Process

This is a binary classification problem and I am using two ensemble methods to predict if a patient has diabetes or not. The two models are Random Forest Classifier and eXtreme Gradient Boosting. Their performances will be compared using evaluation metrics like precision, recall, f1 score and the AUC ROC score.

### The data was explored and cleaned it from inappropriate values. Missing values were disguised as 0. These values were replaced with the mean for `[Insulin]` while for `[Pregnancies]` and `[SkinThickness]`, the zeros were left alone, as this is completely normal. It is not possible for `[Insulin]` to be 0, the person will not survive. Other exploration and preprocessing tasks where carried out as follows:

1. All the columns were numerical datatypes which enabled us carry out statistical tests and plot comprehensive graphs. Histogram plots were used to observe the distribution of the data. Checking for if they follow the Gaussian distribution.

2. Bar plots were used to observe the distribution of the label for each feature. Violin plots were also used to further view the spread of the distribution and box plots were included for observation of the distribution's density and quartile values.

3. Box plots were further used to visualize all the outliers spread for each feature per target label. Pair plot was used to visualize the linear relationship between all the features.

4. Clipping was used to treat outliers in the dataset. Clipping was the choice for treating outliers because the dataset is quite small and the outliers are not evenly spread across the features. Hence, removing any would result in data quality issues and the model would not be properly trained. The clipping was done by calculating the mean and standard deviation as the upper and lower limit.

5. Observed class imbalance in the target and implemented the `stratify` option in the `train_test_split` preprocessing function to take care of the imbalance. The dataset does not have duplicates.

6. The data was then normalized using the `RobustScaler` function. This is because most of the features did not follow a Gaussian distribution.

7. The data set were split for training and validation using the `train_test_split` function. The ratio was 80:20 in favor of the training data.

## Results
The Random Forest Classifier model was trained and evaluated. After hyperparameter tuning, the model had evaluation metrics:
- Precision -- 67.44%
- Recall -- 53.7%
- F1 Score -- 59.79%

And an AUC ROC score of 70%.

The eXtreme Gradient Boost model was trained and evaluated. After hyperparameter tuning, the model had evaluation metrics:
- Precision -- 75.76%
- Recall -- 46.3%
- F1 Score -- 57.47%

And an AUC ROC score of 82.39%.

### Key takeaways
- The evaluation means that the Random Forest model does a better job at correctly predicting positive diabetes cases (true positives) compared to the XGBoost model. 
- The precision and AUC score of the XGBoost is higher than that of the Random Forest. This means that overall, the XGBoost will have maximum performance compared to Random Forest.
- The final decision on which model is better will depend on the intention of the client. If the client wants to minimize false negatives (prioritizing precision) then XGBoost is the better model. If the client wants to prioritize recall (minimize false positives), then Random Forest is the better model.

## Challenges 
 - Not enough time to thoroughly think through the problem formulation and be creative. The estimate given on Compass was completely misleading as the Rubrics requirements are very comprehensive.
 - Everything was done in a rush and there was no time to properly think and intepret the model results and outputs.
 - The dataset is quite small and the preferred model will cannot be deployed because it has not been trained on enough data.

## Future Goals
Investigate the reason why the XGBoost has a lower recall compared to the Random Forest. Investigate the data more in search of possible discrepancies or loopholes.
