
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())
sale_column=data.SalePrice
print(sale_column.head())


# In[ ]:


column_of_interest=data[['YearBuilt','YrSold']]
print(column_of_interest.describe())


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
y=data.SalePrice
sale_predictors=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=data[sale_predictors]
my_model=DecisionTreeRegressor()
my_model.fit(X,y)
print("Making prediction for 5 sale prices")
print(X.head())
print("The predictions are")
print(my_model.predict(X.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predicted_price=my_model.predict(X);
mean_absolute_error(y,predicted_price)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
my_model=DecisionTreeRegressor()
my_model.fit(train_X,train_y)
sales_prediction=my_model.predict(val_X)
print(mean_absolute_error(val_y,sales_prediction))


# In[ ]:


import pandas as pd
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
sale_predictors=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=data[sale_predictors]
y=data.SalePrice
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
def get_mae(max_leaf_nodes,predictors_train,predictors_val,targ_train,targ_val):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(predictors_train,targ_train)
    preds_val=model.predict(predictors_val)
    mae=mean_absolute_error(targ_val,preds_val)
    return mae
for max_leaf_nodes in [5,50,500,5000]:
    my_mae=get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_model=RandomForestRegressor()
forest_model.fit(train_X,train_y)
melb_preds=forest_model.predict(val_X)
print(mean_absolute_error(melb_preds,val_y))


# In[ ]:


import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 2, 3], 
                   'B': [1, 2, np.nan], 
                   'C': [4, 5, 6], 
                   'D': [np.nan, np.nan, np.nan]})

df
[col for col in df.columns if df[col].isnull().any()]


# In[ ]:


import pandas as pd
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
[col for col in data.columns if data[col].isnull().any()]


# In[ ]:


#code to print mean_absolute_error by dropping columns with missing values
import pandas as pd

# Load data
main_file_path = '../input/train.csv'
melb_data = pd.read_csv(main_file_path)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.SalePrice
melb_predictors = melb_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# In[ ]:


#code to print mean_absolute_error by imputer
from sklearn.preprocessing import Imputer
my_imputer=Imputer()
imputed_X_train=my_imputer.fit_transform(X_train)
imputed_X_test=my_imputer.fit_transform(X_test)
print("Mean Absolute error from imputation")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# In[ ]:


melb_predictors.dtypes.sample(10)


# In[ ]:


import pandas as pd
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)
target=train_data.SalePrice
cols_with_missing=[col for col in train_data.columns
                               if train_data[col].isnull().any()]
candidate_train_predictors=train_data.drop(['Id','SalePrice']+cols_with_missing,axis=1)
candidate_test_predictors=test_data.drop(['Id']+cols_with_missing,axis=1)
#finding low_cardinality cols
low_cardinality_cols=[cname for cname in candidate_train_predictors.columns if
                                      candidate_train_predictors[cname].nunique()<10 and
                                      candidate_train_predictors[cname].dtype=='object']
numeric_cols=[cname for cname in candidate_train_predictors.columns if
                              candidate_train_predictors[cname].dtype in ['int64','float64']]
my_cols=low_cardinality_cols+numeric_cols
train_predictors=candidate_train_predictors[my_cols]
test_predictors=candidate_test_predictors[my_cols]
one_hot_encoded_training_predictors=pd.get_dummies(train_predictors)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

