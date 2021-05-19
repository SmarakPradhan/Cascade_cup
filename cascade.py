import numpy as np
import pandas as pd
import seaborn as sns
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from catboost import Pool, CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold

df_train=pd.read_csv("train_age_dataset.csv")
df_test=pd.read_csv("test_age_dataset.csv")
df_train=df_train.rename(columns={"Unnamed: 0":"Unnamed"})
df_test=df_test.rename(columns={"Unnamed: 0":"Unnamed"})
df_train=df_train.sort_index(axis=0)
df_test=df_test.sort_index(axis=0)

df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
x=df_train.drop(['age_group','userId'],axis=1)
y=df_train['age_group'].astype(int)

features = x.columns
numeric = ['Unnamed', 'following_rate', 'followers_avg_age',
       'following_avg_age', 'max_repetitive_punc',
       'num_of_hashtags_per_action', 'emoji_count_per_action',
       'punctuations_per_action', 'number_of_words_per_action',
       'avgCompletion', 'avgTimeSpent', 'avgDuration', 'avgComments',
       'creations', 'content_views', 'num_of_comments',
       'weekends_trails_watched_per_day', 'weekdays_trails_watched_per_day',
       'slot1_trails_watched_per_day', 'slot2_trails_watched_per_day',
       'slot3_trails_watched_per_day', 'slot4_trails_watched_per_day',
       'avgt2']
cat=['tier','gender']
scaler = MinMaxScaler()
x_numeric_std = pd.DataFrame(data=scaler.fit_transform(x[numeric]), columns=numeric)
x_std = pd.merge(x_numeric_std, x[cat], left_index=True, right_index=True)

test=df_test.drop(['userId'],axis=1)
test_numeric_std = pd.DataFrame(data=scaler.fit_transform(test[numeric]), columns=numeric)
test_std = pd.merge(test_numeric_std, test[cat], left_index=True, right_index=True)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, random_state=0,test_size=0.20,stratify=y)
#Selection
# import and create the VarianceThreshold object.
from sklearn.feature_selection import VarianceThreshold
vs_constant = VarianceThreshold(threshold=0)

# select the numerical columns only.
numerical_x_train = x_train[numeric]

# fit the object to our data.
vs_constant.fit(numerical_x_train)

# get the constant colum names.
constant_columns = [column for column in numerical_x_train.columns if column not in numerical_x_train.columns[vs_constant.get_support()]]  

# detect constant categorical variables.
constant_cat_columns = [column for column in x_train[cat] if (len(x[column].unique()) ==1)]
# conctenating the two lists.
all_constant_columns = constant_cat_columns + constant_columns

# drop the constant columns
x_train.drop(labels=all_constant_columns, axis=1, inplace=True)
x_train.drop(labels=all_constant_columns, axis=1, inplace=True)

# make a threshold for quasi constant.
threshold = 0.98

# create empty list
quasi_constant_feature = []

# loop over all the columns
for feature in x_train.columns:

    # calculate the ratio.
    predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]
    
    # append the column name if it is bigger than the threshold
    if predominant >= threshold:
        quasi_constant_feature.append(feature)   
        
print(quasi_constant_feature)

# drop the quasi constant columns
x_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
x_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)

# transpose the feature matrice
train_features_T = x_train.T

# print the number of duplicated features
print(train_features_T.duplicated().sum())

# select the duplicated features columns names
duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

# drop those columns
x_train.drop(labels=duplicated_columns, axis=1, inplace=True)
x_test.drop(labels=duplicated_columns, axis=1, inplace=True)

# creating set to hold the correlated features
corr_features = set()

# create the correlation matrix (default to pearson)
corr_matrix = x_train.corr()

# optional: display a heatmap of the correlation matrix
plt.figure(figsize=(11,11))
sns.heatmap(corr_matrix)

for i in range(len(corr_matrix .columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
            
x_train.drop(labels=corr_features, axis=1, inplace=True)
x_test.drop(labels=corr_features, axis=1, inplace=True)





#LGBM

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

params= {'learning_rate':0.1,
         'boosting_type':'gbdt',
         'colsample_bytree':0.7,
        'importance_type':'split', 
        'max_depth':10,
         'min_child_samples':10, 
         'n_estimators':5000,
         'n_jobs':-1,  
         'objective':'multiclass',
         'random_state':1, 
         'reg_alpha':1.6, 
         'reg_lambda':1.1,
         'subsample':0.7,
         'min_data_in_leaf': 55,
         'num_classes':4}

lgb_clf = lgb.LGBMClassifier(**params)
lgb_clf.fit(x_train,y_train,eval_set=(x_test,y_test),eval_metric='multi_error',early_stopping_rounds=100,categorical_feature=cat)

accuracy_score(y_test, lgb_clf.predict(x_test))

f1_score(y_test, lgb_clf.predict(x_test),average='weighted')