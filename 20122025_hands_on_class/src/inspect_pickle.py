#!/usr/bin/env python3
import sys, os
import pickle
from pprint import pprint
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else 'best_model_HR_analysis.pkl'
if not os.path.isabs(p):
    p = os.path.join(os.getcwd(), p)

print('Inspecting:', p)
try:
    print('Size (bytes):', os.path.getsize(p))
except Exception as e:
    print('Could not get size:', e)

obj = None
# Try pickle then joblib
try:
    with open(p, 'rb') as f:
        obj = pickle.load(f)
    print('Loaded with pickle.load()')
except Exception as e:
    print('pickle.load failed:', repr(e))
    try:
        import joblib
        obj = joblib.load(p)
        print('Loaded with joblib.load()')
    except Exception as e2:
        print('joblib.load failed:', repr(e2))

if obj is None:
    print('Could not load object from file.')
    sys.exit(2)

print('Type:', type(obj))

# Summarize common types
try:
    import pandas as pd
except Exception:
    pd = None

if pd is not None and isinstance(obj, pd.DataFrame):
    print('Pandas DataFrame shape:', obj.shape)
    print('Columns:')
    pprint(list(obj.columns))
elif isinstance(obj, dict):
    print('Dict keys:')
    pprint(list(obj.keys()))
elif isinstance(obj, (list, tuple)):
    print('Sequence length:', len(obj))
    if len(obj) > 0:
        print('First element type:', type(obj[0]))
else:
    # Try scikit-learn estimator
    try:
        from sklearn.base import BaseEstimator
        if isinstance(obj, BaseEstimator):
            print('scikit-learn estimator:', obj.__class__)
            try:
                print('get_params():')
                pprint(obj.get_params())
            except Exception as e:
                print('Could not call get_params():', e)
        else:
            # If it's a NumPy array, give more details
            if isinstance(obj, np.ndarray):
                print('NumPy ndarray dtype:', obj.dtype)
                try:
                    print('ndarray shape:', obj.shape)
                except Exception:
                    pass
                try:
                    print('ndarray ndim:', obj.ndim)
                except Exception:
                    pass
                if obj.ndim == 1:
                    print('ndarray length:', obj.size)
                    print('Elements:')
                    for i, v in enumerate(obj):
                        print(f'[{i}] {v}')
                else:
                    print('Sample rows (first 5):')
                    try:
                        pprint(obj[:5])
                    except Exception:
                        print(repr(obj)[:1000])
            else:
                # Generic repr summary
                print('repr(obj)[:1000] =')
                s = repr(obj)
                print(s[:1000])
    except Exception:
        print('Fallback repr(obj)[:1000] =')
        print(repr(obj)[:1000])

# If estimator has predict, show signature
try:
    import inspect
    if hasattr(obj, 'predict'):
        sig = inspect.signature(obj.predict)
        print('predict signature:', sig)
except Exception:
    pass

print('\nInspection complete.')
