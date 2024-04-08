import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

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
data['windgust'] = data['windgust'].fillna(
    0)  # Assuming missing wind gusts are 0

# Filling missing values or removing unnecessary columns


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

# Define the model with the optimized parameters
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

# Train the model
optimized_model.fit(X_train, y_train, eval_set=[
                    (X_val, y_val)], early_stopping_rounds=50, verbose=False)

# Predict and evaluate on the test set
predictions = optimized_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')
