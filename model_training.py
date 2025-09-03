# import pandas as pd
# from assets_data_prep import prepare_data  # <-- add this import

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# import joblib

# # Load train.csv
# train_df = pd.read_csv('train.csv')

# # Preprocess
# train_df = prepare_data(train_df)

# y_train = train_df['price']
# X_train = train_df.drop(columns='price')

# categorical_cols = ['property_type', 'neighborhood']
# numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer

# numeric_transformer = SimpleImputer(strategy='mean')

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(transformers=[
#     ('num', numeric_transformer, numeric_cols),
#     ('cat', categorical_transformer, categorical_cols)
# ])

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
# ])

# param_distributions = {
#     'regressor__max_depth': randint(3, 30),
#     'regressor__min_samples_split': randint(2, 20),
#     'regressor__min_samples_leaf': randint(1, 20),
#     'regressor__max_features': ['sqrt', 'log2', None]
# }

# random_search = RandomizedSearchCV(
#     model,
#     param_distributions=param_distributions,
#     n_iter=10,
#     cv=10,
#     scoring='neg_root_mean_squared_error',
#     random_state=42,
# )

# random_search.fit(X_train, y_train)

# best_model = random_search.best_estimator_
# joblib.dump(best_model, 'trained_model.pkl')
# print("Model trained and saved as trained_model.pkl")



import pandas as pd
from assets_data_prep import prepare_data
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
import numpy as np


# Load train.csv
train_df = pd.read_csv('train.csv')

# Preprocess
train_df = prepare_data(train_df)
y_train = train_df['price']
X_train = train_df.drop(columns='price')

# Define columns by type
categorical_cols = ['property_type','neighborhood']
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

# Numeric pipeline: impute then scale
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute then one-hot
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline with ElasticNet
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', ElasticNet(random_state=42))
])

# Parameter distributions for RandomizedSearch
param_dist = {
    'model__alpha': np.logspace(-3, 2, 100),  # 0.001 to 100
    'model__l1_ratio': np.linspace(0, 1, 100)
}
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=10,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)

en_model = search.best_estimator_
joblib.dump(en_model, 'trained_model.pkl')
print("Model trained and saved as trained_model.pkl")