import pandas as pd
import numpy as np
from numericConversion import convert_to_numeric
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

#gett eh data od automobile dataset
auto_df = pd.read_csv("../DataSet/Automobile_data.csv")

print("Data loaded to the Pandas Dataframe scuccessfully")

#check for Numeric values in the dataframe and convert to Nuric wherever possible
for column in auto_df.columns:    
    if convert_to_numeric(auto_df[column]):
      auto_df[column] = pd.to_numeric(auto_df[column])

print("Dataframe columns checked and converted to Numeric wherever possible")

#segregate Numeric and Non-Numeric columns
numeric_columns=auto_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns=auto_df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric Columns: {numeric_columns}")
print(f"Categorical Columns: {categorical_columns}")

#impute teh missing vlaues in Numerical columns with mean of the column
for col in numeric_columns:
    auto_df[col].fillna(auto_df[col].mean(), inplace=True)

# convert categorical columns to 'category' dtype and having it unordered to avoid any biasing during model training
for col in categorical_columns:
    auto_df[col] = auto_df[col].astype('category').cat.as_unordered()
    auto_df[col] = auto_df[col].fillna(auto_df[col].mode())  # Fill NaN with mode
    auto_df[col] = auto_df[col].cat.codes  # Accessing cat.codes to ensure categories are set

print("Categorical columns converted to 'category' dtype " \
"successfully and na values are filled with mode of the category" \
" and converted to numerical codes")    

#price column is the target variable so it should have any missing values or NaN values
#drop tehNaN rows from the dataframe
auto_df.dropna(subset=['price'], inplace=True)

#split the data into features and target variable
X = auto_df.drop('price', axis=1)
y = auto_df['price']

print("Feature set and Target variable created successfully")

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Data split into training and testing sets successfully")

# Create and train the Decision Tree Regressor model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(x_train, y_train)

print("Decision Tree Regressor model trained successfully")

# Make predictions
y_pred = dt_regressor.predict(x_train)

print("Predictions made on the test set successfully")

#Model Evaluation
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#predict on test data
y_test_pred = dt_regressor.predict(x_test)

print("Predictions made on the test set successfully")

#Model Evaluation on test data
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test Mean Squared Error: {mse_test}")
print(f"Test R^2 Score: {r2_test}")

#Do the same prediction with Linear Regression model for comparison
linear_regressor = LinearRegression()

linear_regressor.fit(x_train, y_train)

y_train_pred_lr = linear_regressor.predict(x_train)
print("Linear Regression model trained and predictions made on the training set successfully")

#Model Evaluation
mse_lr = mean_squared_error(y_train, y_train_pred_lr)
r2_lr = r2_score(y_train, y_train_pred_lr)
print(f"Linear Regression Mean Squared Error Train: {mse_lr}")    
print(f"Linear Regression R^2 Score Train: {r2_lr}")

y_test_pred_lr = linear_regressor.predict(x_test)
print("Linear Regression model trained and predictions made on the test set successfully")

#Model Evaluation on test data for Linear Regression
mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)
r2_test_lr = r2_score(y_test, y_test_pred_lr)
print(f"Linear Regression Test Mean Squared Error: {mse_test_lr}")
print(f"Linear Regression Test R^2 Score: {r2_test_lr}")

