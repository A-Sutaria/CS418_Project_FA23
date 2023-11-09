import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
zhvi_data = pd.read_csv("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

# Impute missing values using the median
imputer = SimpleImputer(strategy='median')
model_data_numeric = zhvi_data.select_dtypes(include=[np.number])  # Select only numeric columns
model_data_imputed = imputer.fit_transform(model_data_numeric)
model_data_imputed_df = pd.DataFrame(model_data_imputed, columns=model_data_numeric.columns)

# Prepare the historical ZHVI data for modeling
# We'll use the past 12 months of ZHVI values as features to predict the next month's ZHVI value
X = model_data_imputed_df.iloc[:, :-13].values
y = model_data_imputed_df.iloc[:, -13].values  # We use -13 to predict the ZHVI 12 months ahead

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor()

# Train the model on the scaled dataset
gbr.fit(X_train_scaled, y_train)

# Predict the ZHVI values on the test set using the trained model
y_pred_gbr = gbr.predict(X_test_scaled)

# Evaluate the model using the test set
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

print(mae_gbr, rmse_gbr, r2_gbr)
