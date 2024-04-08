# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error

# # Load the dataset
# file_path = 'Bergstrom International A... 2021-07-01 to 2024-03-17.csv'
# data = pd.read_csv(file_path)

# # Data Preprocessing
# data['datetime'] = pd.to_datetime(data['datetime'])
# data['Year'] = data['datetime'].dt.year
# data['Month'] = data['datetime'].dt.month
# data['Day'] = data['datetime'].dt.day
# data['DayOfYear'] = data['datetime'].dt.dayofyear
# data['WeekOfYear'] = data['datetime'].dt.isocalendar().week
# data['windgust'] = data['windgust'].fillna(
#     0)  # Assuming missing wind gusts are 0

# # Filling missing values or removing unnecessary columns


# # Feature selection
# features_to_include = ['humidity', 'windgust',
#                        'cloudcover', 'tempmin', 'tempmax']
# X = data[['Year', 'Month', 'Day', 'DayOfYear',
#           'WeekOfYear'] + features_to_include]
# y = data['tempmax']

# # Scaling the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Splitting the dataset
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, random_state=42)

# # Define the model with the optimized parameters
# optimized_model = XGBRegressor(
#     n_estimators=656,
#     max_depth=5,
#     learning_rate=0.24057515827045411,
#     colsample_bytree=0.9500983814768778,
#     subsample=0.27848270266263386,
#     reg_alpha=0.3728777359951313,
#     reg_lambda=1.9881262022259722,
#     random_state=42
# )

# # Train the model
# optimized_model.fit(X_train, y_train, eval_set=[
#                     (X_val, y_val)], early_stopping_rounds=50, verbose=False)

# # Predict and evaluate on the test set
# predictions = optimized_model.predict(X_test)
# mse = mean_squared_error(y_test, predictions)
# print(f'MSE: {mse}')

from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = 'Bergstrom International A... 2021-07-01 to 2024-03-17.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['datetime'] = pd.to_datetime(data['datetime'])
data['Year'] = data['datetime'].dt.year
data['Month'] = data['datetime'].dt.month
data['Day'] = data['datetime'].dt.day
data['DayOfYear'] = data['datetime'].dt.dayofyear
data['WeekOfYear'] = data['datetime'].dt.isocalendar().week
data['windgust'] = data['windgust'].fillna(0)  # Fill missing wind gusts with 0

# Feature selection
features_to_include = ['humidity', 'windgust',
                       'cloudcover', 'tempmin', 'tempmax']
X = data[['Year', 'Month', 'Day', 'DayOfYear',
          'WeekOfYear'] + features_to_include]
y = data['tempmax']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Define and train the model
optimized_model = XGBRegressor(
    n_estimators=656,
    max_depth=5,
    learning_rate=0.24057515827045411,
    colsample_bytree=0.9500983814768778,
    subsample=0.27848270266263386,
    reg_alpha=0.3728777359951313,
    reg_lambda=1.9881262022259722,
    random_state=42
)
optimized_model.fit(X_train, y_train, eval_set=[
                    (X_val, y_val)], early_stopping_rounds=50, verbose=False)

# Prepare today's data for prediction
today = datetime.now()
df_today = pd.DataFrame({
    'Year': [today.year],
    'Month': [today.month],
    'Day': [today.day],
    'DayOfYear': [today.timetuple().tm_yday],
    'WeekOfYear': [today.isocalendar()[1]],
    'humidity': [np.nan],  # Placeholder, replace with actual data if available
    'windgust': [0],       # Assuming missing wind gusts are 0
    # Placeholder, replace with actual data if available
    'cloudcover': [np.nan],
    # Placeholder, replace with actual data if available
    'tempmin': [np.nan],
    # Placeholder, it's what we're predicting but needs to be in the input
    'tempmax': [np.nan]
})

# Fill missing values for `df_today` as per your dataset's requirements
# For example, df_today['humidity'].fillna(value, inplace=True)

# Scale df_today using the same scaler used for the training data
X_today_scaled = scaler.transform(df_today)

# Predict the max temperature for today
today_max_temp_pred = optimized_model.predict(X_today_scaled)
print(
    f'Predicted max temperature for today: {round(today_max_temp_pred[0])}°F')

# # Define parameter grid
# param_grid = {
#     'n_estimators': [100, 500, 1000],
#     'max_depth': [3, 6, 10],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1],
#     'colsample_bytree': [0.8, 1],
# }

# # Initialize the XGBRegressor
# xgb = XGBRegressor(random_state=42)

# # Initialize the GridSearchCV
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
#                            cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# # Fit GridSearchCV
# grid_search.fit(X_train, y_train)

# # Best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", np.sqrt(-grid_search.best_score_))

# # Use the best estimator for further predictions
# best_model = grid_search.best_estimator_

# # Predictions
# predictions = best_model.predict(X_test)
# mse = mean_squared_error(y_test, predictions)
# print(f'MSE: {mse}')
