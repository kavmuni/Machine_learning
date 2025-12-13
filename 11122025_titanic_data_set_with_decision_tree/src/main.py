# YOutube to check Gini, Entropy - https://www.youtube.com/watch?v=jnV4W3RvVCE
# import the necessary libraries
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# load the train data and test data into dataframes
titanic_train_df = pd.read_csv('../DataSet/train.csv')
titanic_test_df = pd.read_csv('../DataSet/test.csv')

print("Train and Test Data sets loaded from CSV file into Dataframe successfully")

# Check the data imbalance in the target variable 'Survived'
#print(titanic_train_df['Survived'].value_counts(normalize=True), titanic_train_df['Survived'].value_counts())

# Model seems to be biased towards the majority class (not survived)
# We will handle this imbalance during model training using class_weight parameter in Logistic Regression

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
"""For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' 
or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object."""

print("Data preprocessing or Impuation Completed Successfully")

# Check if all missing values are handled
#print(titanic_train_df.isnull().sum())
#print(titanic_test_df.isnull().sum())

# try to proceed with undersampling the majority class to handle data imbalance
categorical_features = ['Sex', 'Cabin_class', 'Is_alone']
numerical_cols_without_survived = ['Parch', 'Fare']
all_model_features = numerical_cols_without_survived + categorical_features

print("Independent Features extracted manually from the dataset")

#Encoding the categorical features and scaling the numerical features
numerical_transformer = Pipeline(
                                steps=[
                                      ("scaler", StandardScaler()),
                                      ("imputer", SimpleImputer(strategy = 'median')) 
                                      ]
                                )

print("Numerical Transformer Created Successfully")

# have the categorical transformer as the categories
"""for cat in categorical_features:
    for df in [titanic_train_df, titanic_test_df]:
        df[cat] = df[cat].astype('category').cat.as_unordered()
        df[cat] = df[cat].fillna(df[cat].mode()[0])
        df[cat] = df[cat].cat.codes"""

categorical_transformer = Pipeline(
                                  steps=[
                                        ("imputer", SimpleImputer(strategy='most_frequent')),
                                        ("onehot", OneHotEncoder(handle_unknown='ignore', drop='first'))  
                                        ]
                                  )

print("Categorical Transformer Created Successfully")

# initialize X and y
X = titanic_train_df[all_model_features]
y = titanic_train_df['Survived']

print("Feature set X and Target variable y created successfully")

# Initialize the RandomUnderSampler
rus = RandomUnderSampler(random_state = 42)

# do the undersampling on X and y
X_resampled, y_resampled = rus.fit_resample(X, y)

print("Random Undersampling completed successfully")
#print(X.shape, y.shape)
#print(y.value_counts(normalize=True), y_resampled.value_counts())
#print(X_resampled.shape, y_resampled.shape)

# preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('numerical', numerical_transformer, numerical_cols_without_survived),
                                               ('categorical', categorical_transformer, categorical_features)])
# Categorical features are already encoded above
print("Preprocessor Created Successfully")

# Split the data into train and test sets
train_X, test_X, train_y, test_y = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42)

print("Train Test Split Completed Successfully")

Decision_tree_classifier = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=1, min_samples_split=2, random_state=42)

# Create the Decision Tree Classifier model pipeline
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", Decision_tree_classifier)])
print("Model Pipeline Created Successfully")

# Fit the model pipeline to the training data
model_pipeline.fit(train_X, train_y)

print("Model fitted/Transformmed to the training data successfully")

# Make predictions on the training data
train_pred = model_pipeline.predict(train_X)
print('*************TRAIN***************')
print(classification_report(train_y, train_pred))
print(confusion_matrix(train_y, train_pred))
print(accuracy_score(train_y, train_pred))
print("The F1Score using DT", f1_score(train_y, train_pred, average='macro'))

# make predictions on the test data
test_pred = model_pipeline.predict(test_X)
print('*************TEST***************')
print(classification_report(test_y, test_pred))
print(confusion_matrix(test_y, test_pred))
print(accuracy_score(test_y, test_pred))
print("The F1Score using DT", f1_score(test_y, test_pred, average='macro'))
print('****************************')
# Make predictions on the actual test data provided
actual_test_X = titanic_test_df[all_model_features]
actual_test_pred = model_pipeline.predict(actual_test_X)
titanic_test_df['Survived'] = actual_test_pred
submission_csv = titanic_test_df[['PassengerId', 'Survived']]
submission_csv.to_csv('submission.csv', index=False)
print('****************************')

# find the number of branches in the decision tree
dt_estimator = model_pipeline.named_steps['model']
n_leaves = dt_estimator.get_n_leaves()
print(f"Number of leaves in the decision tree: {n_leaves}")

#Find the depth of the decision tree
depth = dt_estimator.get_depth()
print(f"Depth of the decision tree: {depth}")

# find Gini importance of each feature
importances = dt_estimator.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("Feature importances (Gini importance):")
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")

# inspect root node stats for fitted tree estimator
feature_names = list(preprocessor.get_feature_names_out())

# tree internals
tree_ = dt_estimator.tree_
root = 0
fidx = tree_.feature[root]
feature = feature_names[fidx]
thr = tree_.threshold[root]
left = tree_.children_left[root]
right = tree_.children_right[root]
n_total = tree_.n_node_samples[root]
imp_parent = tree_.impurity[root]
n_left = tree_.n_node_samples[left]
n_right = tree_.n_node_samples[right]
imp_left = tree_.impurity[left]
imp_right = tree_.impurity[right]

impurity_reduction = imp_parent - (n_left/n_total)*imp_left - (n_right/n_total)*imp_right

print("root feature index:", fidx)
print("root feature name:", feature)
print("threshold:", thr)
print("samples:", n_total, " left:", n_left, " right:", n_right)
print("impurity parent:", imp_parent)
print("impurity left:", imp_left, " impurity right:", imp_right)
print("impurity reduction (gain):", impurity_reduction)

"""
# Grid search or hyperparameter tuning can be done here to improve the model further
# Define the parameter grid for GridSearchCV
param_grid = {
    'model__max_depth': [3, 5, 7, 10, 15, None],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 4],
    'model__criterion': ['gini', 'entropy']
}

print("\n" + "="*60)
print("Starting GridSearchCV for Decision Tree Hyperparameter Tuning...")
print("="*60)

# Create GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=model_pipeline,  # Use the pipeline with preprocessor + model
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1_macro',  # Use F1 macro as scoring metric
    n_jobs=-1,  # Use all available processors
    verbose=1
)

# Fit GridSearchCV on training data
grid_search.fit(train_X, train_y)

print("\n" + "="*60)
print("GridSearchCV Results:")
print("="*60)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score (F1 Macro): {grid_search.best_score_:.4f}")

# Make predictions using the best estimator
best_model = grid_search.best_estimator_
best_train_pred = best_model.predict(train_X)

print("\n" + "="*60)
print("Best Model Performance on Training Data:")
print("="*60)
print(classification_report(train_y, best_train_pred))
print(confusion_matrix(train_y, best_train_pred))
print(f"Accuracy: {accuracy_score(train_y, best_train_pred):.4f}")
print(f"F1 Score (Macro): {f1_score(train_y, best_train_pred, average='macro'):.4f}")

# Compare original model vs best tuned model
print("\n" + "="*60)
print("Comparison: Original vs Tuned Decision Tree")
print("="*60)
print(f"Original Model F1 Score: {f1_score(train_y, train_pred, average='macro'):.4f}")
print(f"Tuned Model F1 Score: {f1_score(train_y, best_train_pred, average='macro'):.4f}")
print(f"Improvement: {(f1_score(train_y, best_train_pred, average='macro') - f1_score(train_y, train_pred, average='macro')):.4f}")

# Extract and display tuned model properties
tuned_dt = best_model.named_steps['model']
print(f"\nTuned Tree Depth: {tuned_dt.get_depth()}")
print(f"Tuned Tree Number of Leaves: {tuned_dt.get_n_leaves()}")


#visualizing the tree to check the Gini values
# Visualize the trained decision tree.
# Extract the fitted estimator from the pipeline and compute feature names
dt_estimator = model_pipeline.named_steps['model']
preproc = model_pipeline.named_steps['preprocessor']

# Try to get transformed feature names (works with sklearn >= 1.0)
try:
    feature_names = preproc.get_feature_names_out()
except Exception:
    # Fallback: try to transform a single row and build generic feature names
    try:
        transformed = preproc.transform(train_X.head(1))
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]
    except Exception:
        # As a last resort use the original feature list (may not match one-hot expansion)
        feature_names = all_model_features

# Class names from the fitted estimator
class_names = [str(c) for c in getattr(dt_estimator, "classes_", [0, 1])]

# export_graphviz returns DOT data when out_file=None. Try Graphviz first,
# if Graphviz 'dot' executable is missing fall back to matplotlib's plot_tree.
try:
    dot_data = export_graphviz(dt_estimator,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True,
                               rounded=True,
                               special_characters=True)

    dec_tree_undersampled_graph = graphviz.Source(dot_data)
    # Render to PNG using Graphviz (file will be 'titanic_decision_tree.png')
    dec_tree_undersampled_graph.render('titanic_decision_tree', format='jpeg', cleanup=True)
    try:
        dec_tree_undersampled_graph.view()
    except Exception:
        print("Rendered decision tree to 'titanic_decision_tree.jpeg' (viewing not available).")
except Exception as e:
    # Likely Graphviz 'dot' is not installed or not on PATH. Fall back to matplotlib.
    print(f"Graphviz render failed: {e}")
    print("Falling back to matplotlib plot_tree rendering...")
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(dt_estimator, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
        plt.savefig('titanic_decision_tree_matplotlib.jpeg', bbox_inches='tight')
        plt.close()
        print("Rendered decision tree to 'titanic_decision_tree_matplotlib.jpeg' using matplotlib.")
    except Exception as e2:
        print(f"Matplotlib fallback also failed: {e2}")
"""
print('****************************')

# do the same activities on train data for logisticregression
logistic_model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", LogisticRegression(class_weight='balanced', random_state=42))])

print("Logistic Regression Model Pipeline Created Successfully")

# Fit the logistic regression model pipeline to the training data
logistic_model_pipeline.fit(train_X, train_y)

print("Logistic Regression Model fitted/Transformmed to the training data successfully")

# Make predictions on the training data using logistic regression
train_pred_logistic = logistic_model_pipeline.predict(train_X)
print('*************TRAIN - Logistic Regression ***************')
print(classification_report(train_y, train_pred_logistic))
print(confusion_matrix(train_y, train_pred_logistic))
print(accuracy_score(train_y, train_pred_logistic))
print("The F1Score using Logistic Regression", f1_score(train_y, train_pred_logistic, average='macro'))
print('****************************')

# do the prediction without undersampling to see the difference
# Split the original data into train and test sets
orig_train_X, orig_test_X, orig_train_y, orig_test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Original Train Test Split Completed Successfully")
model_pipeline_dec_imbalance = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", DecisionTreeClassifier(criterion='entropy', random_state=42))])
# Decision Tree without undersampling
model_pipeline_dec_imbalance.fit(orig_train_X, orig_train_y)

print("Decision Tree Model fitted/Transformmed to the original training data successfully")

# Make predictions on the original training data
orig_train_pred = model_pipeline_dec_imbalance.predict(orig_train_X)
print('*************ORIGINAL TRAIN***************')
print(classification_report(orig_train_y, orig_train_pred))
print(confusion_matrix(orig_train_y, orig_train_pred))
print(accuracy_score(orig_train_y, orig_train_pred))
print("The F1Score using DT on original data", f1_score(orig_train_y, orig_train_pred, average='macro'))
print('****************************')


# Fit the logistic regression model pipeline to the original training data
logistic_model_pipeline.fit(orig_train_X, orig_train_y)

print("Logistic Regression Model fitted/Transformmed to the original training data successfully")

# Make predictions on the original training data using logistic regression
orig_train_pred_logistic = logistic_model_pipeline.predict(orig_train_X)
print('*************ORIGINAL TRAIN - Logistic Regression ***************')
print(classification_report(orig_train_y, orig_train_pred_logistic))
print(confusion_matrix(orig_train_y, orig_train_pred_logistic))
print(accuracy_score(orig_train_y, orig_train_pred_logistic))
print("The F1Score using Logistic Regression on original data", f1_score(orig_train_y, orig_train_pred_logistic, average='macro'))
print('****************************')

# group confusion matrix values of all above models for comparison
tn, fp, fn, tp = confusion_matrix(train_y, train_pred).ravel()

print(f"Decision Tree with Undersampling - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

tn_log, fp_log, fn_log, tp_log = confusion_matrix(train_y, train_pred_logistic).ravel()

print(f"Logistic Regression with Undersampling - TP: {tp_log}, TN: {tn_log}, FP: {fp_log}, FN: {fn_log}")

tn_orig, fp_orig, fn_orig, tp_orig = confusion_matrix(orig_train_y, orig_train_pred).ravel()

print(f"Decision Tree without Undersampling - TP: {tp_orig}, TN: {tn_orig}, FP: {fp_orig}, FN: {fn_orig}")

tn_orig_log, fp_orig_log, fn_orig_log, tp_orig_log = confusion_matrix(orig_train_y, orig_train_pred_logistic).ravel()

print(f"Logistic Regression without Undersampling - TP: {tp_orig_log}, TN: {tn_orig_log}, FP: {fp_orig_log}, FN: {fn_orig_log}")


# combine all models into a dictionary for future use
models = {
    "Decision Tree with Undersampling": model_pipeline,
    "Logistic Regression with Undersampling": logistic_model_pipeline,
    "Decision Tree without Undersampling": model_pipeline,
    "Logistic Regression without Undersampling": logistic_model_pipeline
}

print("All Models are created and stored in the dictionary successfully")

