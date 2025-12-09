import pandas as pd
import numpy as np
from numericConversion import convert_to_numeric
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

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

price_correlation = auto_df[numeric_columns].corr()['price']
price_correlation.drop('price', inplace=True)
price_correlation = price_correlation.sort_values(ascending=False)
num_column = []
for rec in price_correlation.items():
  if rec[1] > 0.1:
    num_column.append(rec[0])
    print(f"Column: {num_column}, Correlation: {rec[1]}")
print(f"Columns having correlation more than 0.1 with price column: {num_column}")

#split the data into features and target variable
#X = auto_df.drop('price', axis=1)
X = auto_df[num_column + categorical_columns]
y = auto_df['price']

print("Feature set and Target variable created successfully")

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Data split into training and testing sets successfully")
print("***********************************")
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
print("***********************************")
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
print("***********************************")
# do the same prediction with L2 Regularized Linear Regression (Ridge Regression) for comparison
ridge_regressor = Ridge(alpha=0.01, max_iter=1000)

ridge_regressor.fit(x_train, y_train)

y_train_pred_ridge = ridge_regressor.predict(x_train)

print("Ridge Regression model trained and predictions made on the training set successfully")

#Model Evaluation
mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
r2_ridge = r2_score(y_train, y_train_pred_ridge)
print(f"Ridge Regression Mean Squared Error Train: {mse_ridge}")
print(f"Ridge Regression R^2 Score Train: {r2_ridge}")
y_test_pred_ridge = ridge_regressor.predict(x_test)
print("Ridge Regression model trained and predictions made on the test set successfully")   
#Model Evaluation on test data for Ridge Regression
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
print(f"Ridge Regression Test Mean Squared Error: {mse_test_ridge}")    
print(f"Ridge Regression Test R^2 Score: {r2_test_ridge}")
print("***********************************")
# do the same prediction with L1 Regularized Linear Regression (Lasso Regression) for comparison
lasso_regressor = Lasso(alpha=0.01, max_iter=1000)
lasso_regressor.fit(x_train, y_train)
y_train_pred_lasso = lasso_regressor.predict(x_train)
print("Lasso Regression model trained and predictions made on the training set successfully")

#Model Evaluation
mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
r2_lasso = r2_score(y_train, y_train_pred_lasso)
print(f"Lasso Regression Mean Squared Error Train: {mse_lasso}")
print(f"Lasso Regression R^2 Score Train: {r2_lasso}")

y_test_pred_lasso = lasso_regressor.predict(x_test)
print("Lasso Regression model trained and predictions made on the test set successfully")       

#Model Evaluation on test data for Lasso Regression
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)  
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
print(f"Lasso Regression Test Mean Squared Error: {mse_test_lasso}")    
print(f"Lasso Regression Test R^2 Score: {r2_test_lasso}")
print("***********************************")
#use grid search to find the best hyperparameters for Lasso Regression
param_grid = {'alpha':np.linspace(0.001, 0.1, 20)}
grid_search = GridSearchCV(Lasso(max_iter=1000), param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Grid Search for Lasso Regression completed successfully")  
print(f"Best parameters for Lasso Regression: {grid_search.best_params_}")
best_lasso_model = grid_search.best_estimator_
y_test_pred_best_lasso = best_lasso_model.predict(x_test)
print("Best Lasso Regression model predictions made on the test set successfully")
#Model Evaluation on test data for best Lasso Regression
mse_test_best_lasso = mean_squared_error(y_test, y_test_pred_best_lasso)
r2_test_best_lasso = r2_score(y_test, y_test_pred_best_lasso)
print(f"Best Lasso Regression Test Mean Squared Error: {mse_test_best_lasso}")    
print(f"Best Lasso Regression Test R^2 Score: {r2_test_best_lasso}")
print("***********************************")
#use grid search to find the best hyperparameters for Ridge Regression
grid_search_ridge = GridSearchCV(Ridge(max_iter=1000), param_grid, cv=5)
grid_search_ridge.fit(x_train, y_train)
print("Grid Search for Ridge Regression completed successfully")  
print(f"Best parameters for Ridge Regression: {grid_search_ridge.best_params_}")
best_ridge_model = grid_search_ridge.best_estimator_
y_test_pred_best_ridge = best_ridge_model.predict(x_test)
print("Best Ridge Regression model predictions made on the test set successfully")
#Model Evaluation on test data for best Ridge Regression
mse_test_best_ridge = mean_squared_error(y_test, y_test_pred_best_ridge)
r2_test_best_ridge = r2_score(y_test, y_test_pred_best_ridge)
print(f"Best Ridge Regression Test Mean Squared Error: {mse_test_best_ridge}")    
print(f"Best Ridge Regression Test R^2 Score: {r2_test_best_ridge}")
print("***********************************")
# combine all the model results into a dataframe for comparison
model_comparison = pd.DataFrame({
    'Model': ['Decision Tree Regressor', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Best Lasso Regression', 'Best Ridge Regression'],
    'MSE_Train': [round(mse, 2), round(mse_lr, 2), round(mse_ridge, 2), round(mse_lasso, 2), round(mse_lasso, 2), round(mse_ridge, 2)],
    'R2_Train': [r2, r2_lr, r2_ridge, r2_lasso, r2_lasso, r2_ridge],
    'MSE_Test': [round(mse_test, 2), round(mse_test_lr, 2), round(mse_test_ridge, 2), round(mse_test_lasso, 2), round(mse_test_best_lasso, 2), round(mse_test_best_ridge, 2)],
    'R2_Test': [r2_test, r2_test_lr, r2_test_ridge, r2_test_lasso, r2_test_best_lasso, r2_test_best_ridge]
})
print("Model Comparison:")
print(model_comparison)
print("***********************************")