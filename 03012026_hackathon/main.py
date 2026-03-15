import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


hr_train_df = pd.read_csv('train.csv')
hr_test_df = pd.read_csv('test.csv')
hr_test_df_copy=hr_test_df.copy()

# convert all the values of both data frame into upper case so all the values are treated same when it comes to unknown data
for df in [hr_train_df, hr_test_df]:
    df.drop_duplicates()
    df.drop(columns=['employee_id'], inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.upper()
    df['education'] = df['education'].fillna(df.groupby(['gender', 'department'])['education'].transform(lambda x: x.mode()[0]))
    df['education'] = df['education'].fillna(df.groupby(['gender'])['education'].transform(lambda x: x.mode()[0]))
    df['previous_year_rating'] = df['previous_year_rating'].fillna(df.groupby(['gender', 'department'])['previous_year_rating'].transform(lambda x: x.mode()[0]))
    df['previous_year_rating'] = df['previous_year_rating'].fillna(df.groupby(['gender'])['previous_year_rating'].transform(lambda x: x.mode()[0]))

X=hr_train_df.drop(columns=['is_promoted'])
y=hr_train_df['is_promoted']

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), #Imputer is NOT needed here as the NULL values are filled above
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), #Imputer is NOT needed here as the NULL values are filled above
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(min_samples_split=2, random_state=42)
# Apply preprocessor to X_train before calling cost_complexity_pruning_path
X_train_processed = preprocessor.fit_transform(X_train)
path = decision_tree.cost_complexity_pruning_path(X_train_processed, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

model_pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", decision_tree)
    ]
)
"""
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
"""
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    model_pipeline=Pipeline(
      steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ]
    )
    model_pipeline.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)