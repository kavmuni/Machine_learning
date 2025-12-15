import pandas as pd
import numpy as np

def sklearn_root_gini(X,y):
  result = []
  y = np.asarray(y)

  classes = np.unique(y)
  #print(f"classes: {classes}")
  n_classes = len(classes)

  class_index = {cls:i for i,cls in enumerate(classes)}
  #print(f"class_index: {class_index}")
  y_encoded = np.array([class_index[i] for i in y])
  #print(X.columns)
  for feature in X.columns:
    values = np.asarray(X[feature])
    order = np.argsort(values)
    #order = np.unique(order)
    values_sorted = values[order]
    y_sorted = y_encoded[order]
    #y_encoded = y_encoded[order]
    #print(f"feature: {feature} - values_sorted: {values_sorted} - order: {order}")
    right_counts = np.bincount(y_encoded, minlength = n_classes).astype(float)
    left_counts = np.zeros(n_classes)

    best_gini = 1.0
    best_threshold = None
    #print(f"values_sorted.dtype: {values_sorted.dtype}")
    for i in range(len(values_sorted) -1):
      cls = y_sorted[i]
      left_counts[cls]+=1
      right_counts[cls]-=1

      if values_sorted[i] == values_sorted[i+1]:
        continue

      threshold = (values_sorted[i] + values_sorted[i+1])/2
      left_total = i+1
      right_total = len(y_sorted)-left_total

      #Gini left
      p_left = left_counts/left_total
      g_left = 1.0-np.sum(p_left**2)

      #Gini right
      p_right = right_counts/right_total
      g_right = 1.0-np.sum(p_right**2)

      #weighted gini after split
      g_split = (left_total / len(y_sorted))*g_left + (right_total / len(y_sorted))*g_right

      if g_split < best_gini:
        best_gini= g_split
        best_threshold = threshold

      result.append((feature, best_gini, best_threshold))
  results = pd.DataFrame(result, columns = ['feature','gini','threshold'])
  return results.sort_values("gini")
