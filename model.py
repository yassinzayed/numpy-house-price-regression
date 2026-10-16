"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    nan_columns = np.where(np.isnan(X).all(axis=0))[0]
    X[:, nan_columns] = np.zeros((X.shape[0], len(nan_columns)))
    X_clean = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X)[i][j]:
                X_clean[i][j] = np.nanmean(X[:, j])
    return X_clean

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)
    iqr = q3-q1
    lower = q1-k*iqr
    upper = q3+k*iqr
    return (lower, upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.where(X < lower, lower, np.where(X > upper, upper, X))

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.column_stack((X, col))
    return X

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    one_hot = np.zeros((len(labels), len(np.unique(labels))))
    for i in range(len(np.unique(labels))):
        indices = np.where(labels == (np.unique(labels)[i]))[0]
        one_hot[indices, i] = 1
    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.where(np.std(X, axis=0)==0, 1.0, np.std(X, axis=0))
    return (mean, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X = (X-mean)/std
    return X

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    X = np.column_stack((np.ones(len(X)), X))
    return X

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    x = np.arange(n_samples)
    np.random.shuffle(x)
    return x

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    train_elements = int(len(indices)*train_ratio)
    val_elements = int(len(indices)*(train_ratio+val_ratio))
    train_idx = indices[:train_elements]
    val_idx = indices[train_elements:val_elements]
    test_idx = indices[val_elements:]
    return (train_idx, val_idx, test_idx)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return (X[indices], y[indices])

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    mae = np.mean(np.abs(y_true-y_pred))
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    ss_res = sum(np.square(y_true-y_pred))
    ss_tot = sum(np.square(y_true-np.mean(y_true)))
    if ss_tot == 0:
        return 0.0
    r_2 = 1-(ss_res/ss_tot)
    return r_2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    return {
        'mean': np.mean(y_true-y_pred),
        'std': np.std(y_true-y_pred),
        'median_abs': np.median(np.abs(y_true-y_pred))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_imp = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imp, k=iqr_k)
    res = clip_columns(X_imp, lower, upper)
    return res

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num = X_num[:, ratio_num_idx]
    denom = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num, denom)
    appended = append_column(X_num, ratio)
    if cat_labels is not None:
        matrix = np.hstack([appended, one_hot_encode(cat_labels)])
        return matrix
    else:
        return appended

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_sampels = X.shape[0]
    shuffled = make_shuffled_indices(n_sampels, seed)
    partioned = partition_indices(shuffled, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, partioned[0])
    X_val, y_val = subset_xy(X, y, partioned[1])
    X_test, y_test = subset_xy(X, y, partioned[2])
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    mean, std = fit_standardizer(splits['X_train'])
    X_train = add_bias_column(apply_standardizer(splits['X_train'], mean, std))
    X_val = add_bias_column(apply_standardizer(splits['X_val'], mean, std))
    X_test = add_bias_column(apply_standardizer(splits['X_test'], mean, std))
    final_dict = {
        'X_train': X_train,
        'y_train': splits['y_train'],
        'X_val': X_val,
        'y_val': splits['y_val'],
        'X_test': X_test,
        'y_test': splits['y_test']
    }
    return (final_dict, mean, std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'residual_summary': residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

