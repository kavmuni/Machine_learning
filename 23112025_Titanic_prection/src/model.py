import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

import joblib

# Loading the train and test data into dataframes
titanic_train_df = pd.read_csv('DataSet/train.csv')
titanic_test_df = pd.read_csv('DataSet/test.csv')

print("Train and Test Data Loaded Successfully")

for df in [titanic_train_df, titanic_test_df]:
  df['Cabin'] = df.groupby(['Pclass', 'Sex'])['Cabin'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
  df['Age'] = df['Age'].apply(lambda x: round(x) if pd.notna(x) else x)
  df['Age'] = df['Age'].apply(lambda x: 1 if x < 1 else x)
  df['Age'].fillna(df.groupby(['Pclass','Sex', 'Parch', 'SibSp'])['Age'].transform('median'), inplace=True)
  df['Age'].fillna(df.groupby(['Pclass','Sex', 'Parch'])['Age'].transform('median'), inplace=True)
  df['Age'].fillna(df.groupby(['Pclass','Sex'])['Age'].transform('median'), inplace=True)
  df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
  df['Fare'].fillna(df.groupby(['Embarked', 'Pclass', 'Cabin'])['Fare'].transform('median'), inplace=True)
  df['Cabin_class'] = df['Pclass'].astype(str) + df['Cabin'].str[0]
  df['Is_alone'] = df.apply(lambda row: 'Y' if row['SibSp'] + row['Parch'] == 0 else 'N', axis=1)
  df['Fare_log'] = df.apply(lambda row: np.log1p(row['Fare']), axis=1)

print("Data preprocessing or Impuation Completed Successfully")

numerical_features = ['SibSp','Parch', 'Survived', 'Fare_log']
categorical_features = ['Sex', 'Cabin_class', 'Is_alone']
numerical_cols_without_survived = ['Parch', 'Fare']

print("Independent Features extracted manually from the dataset")

numerical_transformer = Pipeline(
                                steps=[
                                      ("imputer", SimpleImputer(strategy='median')),
                                      ("scaler", StandardScaler())  
                                      ]
                                )

print("Numerical Transformer Created Successfully")

categorical_transformer = Pipeline(
                                  steps=[
                                        ("imputer", SimpleImputer(strategy='most_frequent')),
                                        ("onehot", OneHotEncoder(handle_unknown='ignore', drop='first'))  
                                        ]
                                  )

print("Categorical Transformer Created Successfully")

preprocessor = ColumnTransformer(
  transformers=[('numerical', numerical_transformer, numerical_cols_without_survived),
                ('categorical_transformer', categorical_transformer, categorical_features)]
)

print("Preprocessor Created Successfully")

model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), 
                                 ("model", LogisticRegression())])

print("Model Pipeline Created Successfully")

X = titanic_train_df[numerical_cols_without_survived + categorical_features]

print("Feature set X created successfully")

y = titanic_train_df['Survived']

print("Feature set y created successfully")

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Train Test Split Completed Successfully")

model_pipeline.fit(train_X, train_y)

print("Model fitted/Transformmed to the training data successfully")

train_pred = model_pipeline.predict(train_X)
print(accuracy_score(train_y, train_pred))
print('****************************')
test_pred = model_pipeline.predict(test_X)
print(accuracy_score(test_y, test_pred))
print('*************TRAIN***************')
print(classification_report(train_y, train_pred))
print('*************TEST***************')
print(classification_report(test_y, test_pred))
print('************TRAIN****************')
print(confusion_matrix(train_y, train_pred))

print('*************TEST***************')
print(confusion_matrix(test_y, test_pred))

test_pred_on_unseen_data = model_pipeline.predict(titanic_test_df[numerical_cols_without_survived + categorical_features])

print("Prediction on unseen test data completed successfully")

# Save the model pipeline to a file

joblib.dump(model_pipeline, 'titanic_model_pipeline.pkl')

print("Model Pipeline saved successfully as titanic_model_pipeline.pkl")