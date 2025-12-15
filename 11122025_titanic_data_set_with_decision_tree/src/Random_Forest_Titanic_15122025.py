import pandas as pd
import numpy as np
import warnings

from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import xgboost as xgb
import sklearn_root_gini
from sklearn.tree import export_graphviz, plot_tree
import graphviz
import matplotlib.pyplot as plt
import joblib
import os



titanic_train_df = pd.read_csv("../DataSet/train.csv")
print("Train Dataset loaded successfully")

titanic_test_df = pd.read_csv("../DataSet/test.csv")
print("Test Dataset loaded successfully")

# imputations and feature engineering
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

print("Imputation on the columns are completed")    

categorical_features = ['Sex', 'Cabin_class', 'Is_alone']
numerical_cols_without_survived = ['Parch', 'Fare']
all_model_features = numerical_cols_without_survived + categorical_features

print("*************Random Forest Classification starts*********************")

numerical_transformer = Pipeline(
                                steps=[
                                      ("scaler", StandardScaler()),
                                      ("imputer", SimpleImputer(strategy = 'median')) 
                                      ]
                                )

categorical_transformer = Pipeline(
                                  steps=[
                                        ("imputer", SimpleImputer(strategy='most_frequent')),
                                        ("onehot", OneHotEncoder(handle_unknown='ignore', drop='first'))  
                                        ]
                                  )

preprocessor = ColumnTransformer(transformers=
                                 [("numerical", numerical_transformer, numerical_cols_without_survived),
                                  ("categorical", categorical_transformer, categorical_features)])

random_forest = RandomForestClassifier()

pipeline_without_model = Pipeline(steps=[("preprocessor", preprocessor)])

X = titanic_train_df[all_model_features]
print("X DF loaded")

y = titanic_train_df["Survived"]
print("y loaded")

pipeline_without_model.fit(X, y)
preprocessor_obj = pipeline_without_model.named_steps['preprocessor']
transformed_X = preprocessor_obj.transform(X)
feature_names_out = preprocessor_obj.get_feature_names_out()
if hasattr(transformed_X, 'toarray'):
    transformed_X = transformed_X.toarray()
transformed_X = pd.DataFrame(transformed_X, columns=feature_names_out)
print(transformed_X.head(5))

train_x, test_x, train_y, test_y = train_test_split(transformed_X, y, test_size=0.2, random_state=42)
pipeline = Pipeline(steps=[("model", random_forest)])
pipeline.fit(train_x, train_y)

print("train values are fitted with Random Forest")

y_train_pred = pipeline.predict(train_x)

acc_score = accuracy_score(train_y, y_train_pred)

print(f"Accuracy Score of the train data with Random Forest -  {acc_score}")

y_test_pred = pipeline.predict(test_x)

acc_score_test_RF = accuracy_score(test_y, y_test_pred)

print(f"Accuracy Score of the test data with Random Forest -  {acc_score_test_RF}")

#print(sklearn_root_gini.sklearn_root_gini(train_x, train_y).drop_duplicates())
#print(f"shape of train_x is: {train_x.shape}")

xgboost = xgb.XGBClassifier()

pipeline_with_xgb = Pipeline(steps=[("model", xgboost)])

pipeline_with_xgb.fit(train_x, train_y)

y_xgb_pred = pipeline_with_xgb.predict(train_x)

acc_score_xgb = accuracy_score(train_y, y_xgb_pred)

print(f"Accuracy Score of the train data with XGB -  {acc_score_xgb}")

y_test_pred_xgb = pipeline_with_xgb.predict(test_x)

acc_score_test_xgb = accuracy_score(test_y, y_test_pred_xgb)

print(f"Accuracy Score of the test data with XGB -  {acc_score_test_xgb}")

print("*************Random Forest Classification ends*********************")
"""
# Find trees with categorical__Sex_male as root node
print("\n*************Analyzing Random Forest Trees*********************")

rf_model = pipeline.named_steps['model']
feature_names_list = list(feature_names_out)
target_root_feature = 'categorical__Sex_male'

trees_with_target_root = []

trees_dir = "random_forest_trees_sex_root_node"

for i, tree in enumerate(rf_model.estimators_):
    tree_obj = tree.tree_
    root_feature_idx = tree_obj.feature[0]  # feature index at root node
    root_feature_name = feature_names_list[root_feature_idx]
    
    if root_feature_name == target_root_feature:
        trees_with_target_root.append(i)
        threshold = tree_obj.threshold[0]
        n_samples = tree_obj.n_node_samples[0]
        print(f"Tree {i}: Root feature = {root_feature_name}, Threshold = {threshold}, Samples = {n_samples}")
        dot_data = export_graphviz(tree,
                                   out_file=None,
                                   feature_names=list(feature_names_out),
                                   class_names=['Did Not Survive', 'Survived'],
                                   filled=True,
                                   rounded=True,
                                   special_characters=True)
        
        graph = graphviz.Source(dot_data)
        graph.render(f'{trees_dir}/tree_{i}', format='jpeg', cleanup=True)
        print(f"Saved tree {i} using Graphviz: {trees_dir}/tree_{i}.jpeg")
print(f"\nTotal trees with '{target_root_feature}' as root: {len(trees_with_target_root)}")
print(f"Tree indices: {trees_with_target_root}")

print("*************Random Forest Classification ends*********************")



print("\n*************Saving Random Forest Trees*********************")

# Get the trained random forest model from pipeline
rf_model = pipeline.named_steps['model']
n_trees = rf_model.n_estimators

print(f"Random Forest has {n_trees} trees. Saving first 5 trees...")

# Create a directory to store tree visualizations
trees_dir = "random_forest_trees"
if not os.path.exists(trees_dir):
    os.makedirs(trees_dir)

# Save first 5 trees as PNG/JPEG images
for i in range(min(5, n_trees)):
    tree = rf_model.estimators_[i]
    
    # Try using Graphviz first
    try:
        dot_data = export_graphviz(tree,
                                   out_file=None,
                                   feature_names=list(feature_names_out),
                                   class_names=['Did Not Survive', 'Survived'],
                                   filled=True,
                                   rounded=True,
                                   special_characters=True)
        
        graph = graphviz.Source(dot_data)
        graph.render(f'{trees_dir}/tree_{i}', format='jpeg', cleanup=True)
        print(f"Saved tree {i} using Graphviz: {trees_dir}/tree_{i}.jpeg")
    except Exception as e:
        # Fallback to matplotlib
        print(f"Graphviz failed for tree {i}: {e}. Using matplotlib fallback...")
        try:
            plt.figure(figsize=(20, 10))
            plot_tree(tree, feature_names=list(feature_names_out), 
                     class_names=['Did Not Survive', 'Survived'], 
                     filled=True, rounded=True)
            plt.savefig(f'{trees_dir}/tree_{i}_matplotlib.jpeg', bbox_inches='tight')
            plt.close()
            print(f"Saved tree {i} using matplotlib: {trees_dir}/tree_{i}_matplotlib.jpeg")
        except Exception as e2:
            print(f"Failed to save tree {i}: {e2}")

# Save the entire Random Forest model as pickle
model_path = 'random_forest_model.pkl'
joblib.dump(rf_model, model_path)
print(f"\nSaved entire Random Forest model to: {model_path}")

# Save feature names and tree information
info = {
    'n_trees': n_trees,
    'feature_names': list(feature_names_out),
    'class_names': ['Did Not Survive', 'Survived'],
    'train_accuracy': acc_score
}
joblib.dump(info, 'random_forest_info.pkl')
print(f"Saved Random Forest info to: random_forest_info.pkl")
"""
print("*************Random Forest Trees Saved Successfully*********************")

# -----------------------------
# Save XGBoost stumps (trees with max_depth=1)
# -----------------------------
print("\n*************Training XGB stump model and saving stumps*********************")

# Train an XGBoost model constrained to stumps
stump_n_estimators = 50
xgb_stump = xgb.XGBClassifier(max_depth=1, n_estimators=stump_n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_stump.fit(train_x, train_y)

# Directory to save stump images and dumps
xgb_dir = 'xgb_stumps'
os.makedirs(xgb_dir, exist_ok=True)

# Also dump textual representation of each tree
booster = xgb_stump.get_booster()
tree_dump = booster.get_dump(with_stats=True)

for i, tree_text in enumerate(tree_dump):
    try:
        dot_data = xgb.to_graphviz(xgb_stump, num_trees=1)
        dot_data.render(f'{xgb_dir}/tree_{i}', format='jpeg')  # Saves tree.png and opens

        with open(os.path.join(xgb_dir, f'stump_{i}.txt'), 'w') as f:
            f.write(tree_text)
    except Exception as e:
        print(f"Failed to write stump text {i}: {e}")

# Save the XGB stump model and metadata
joblib.dump(xgb_stump, os.path.join(xgb_dir, 'xgb_stump_model.pkl'))
joblib.dump({'n_estimators': stump_n_estimators, 'feature_names': list(feature_names_out)}, os.path.join(xgb_dir, 'xgb_stump_info.pkl'))

print("Saved XGB stumps and model under:", xgb_dir)
