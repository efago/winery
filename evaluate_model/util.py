import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def plot_feature_importance(xgb_model, path):
    """Plots feature importances from the trained model
    
    Parameters
    ----------
    xgb_model: Booster
        XGBoost model to be evaluated
    path: str
        directory where plot should be saved

    Returns
    -------
    None
    """
    ax = xgb.plot_importance(xgb_model)
    ax.figure.tight_layout()
    ax.figure.savefig(path + 'feature_importance.png')
    plt.close()

def plot_regression(test_y, preds, idx, path):
    """Plots regression and scatter plots of labels and prediction
    
    Parameters
    ----------
    test_y: ndarray
        array of labels
    preds: ndarray
        array of predictions
    idx: ndarray
        array of sorted labels indexes
    path: str
        directory where plot should be saved

    Returns
    -------
    None
    """
    sns.regplot(preds[idx], np.sort(test_y))
    plt.plot(range(80, 100), range(80, 100), label='Ideal regression')
    plt.ylabel('Target points')
    plt.xlabel('Predictions')
    plt.ylim(80, 100)
    plt.xlim(80, 100)
    plt.title('Scatterplot of predictions and target')
    plt.legend()
    plt.grid()
    plt.savefig(path + 'regressions.png')
    plt.close()

def plot_distribution(test_y, preds, path):
    """Plots boxplot of labels and prediction values
    
    Parameters
    ----------
    test_y: ndarray
        array of labels
    preds: ndarray
        array of predictions
    path: str
        directory where plot should be saved

    Returns
    -------
    None
    """
    sns.boxplot(x=['Target', 'Prediction'], y=[test_y, preds])
    plt.ylabel('Points')
    plt.title('Distribution of points')
    plt.savefig(path + 'distributions.png')
    plt.close()

def plot_error_cumulative(test_y, preds, idx, rmse, path):
    """Plots cumulative of absolute errors with ascending label values
    
    Parameters
    ----------
    test_y: ndarray
        array of labels
    preds: ndarray
        array of predictions
    idx: ndarray
        array of sorted labels indexes
    rmse: float
        root mean squared error of predictions
    path: str
        directory where plot should be saved.

    Returns
    -------
    None
    """
    errs_y_asc = abs(np.sort(test_y) - preds[idx])
    plt.plot(np.sort(test_y), errs_y_asc.cumsum(), \
        label='RMSE= {:.3f}'.format(rmse))
    plt.xlabel('Target points')
    plt.ylabel('Cumulative error')
    plt.title('Progression of errors with ascending target points')
    plt.legend()
    plt.savefig(path + 'error_cumulative.png')
    plt.close()

def compare_model_baseline(test_y, preds, idx, train_mean, path):
    """Plots cumulative errors of baseline and trained models
    
    Parameters
    ----------
    test_y: ndarray
        array of labels
    preds: ndarray
        array of predictions
    idx: ndarray
        array of sorted labels indexes
    train_mean: float
        baseline predictions which is equal to mean of training labels
    path: str
        directory where plot should be saved.

    Returns
    -------
    None
    """    
    errs_y_asc = (np.sort(test_y) - preds[idx]).cumsum()
    errs_baseline = (np.sort(test_y) - [train_mean]*len(test_y)).cumsum()
    plt.plot(np.sort(test_y), errs_y_asc, label='xgb_pred')
    plt.plot(np.sort(test_y), errs_baseline, label='baseline')
    plt.plot([test_y.mean(), test_y.mean()], [0, errs_baseline.min()], \
        label= 'mean')
    plt.ylabel('Cumulative errors')
    plt.xlabel('Target points')
    plt.title('Comparison of trained and baseline models')
    plt.legend()
    plt.savefig(path + 'comparison.png')
    plt.close()

def plot_error_per_point(test_pred, path):
    """Plots net error per label
    
    Parameters
    ----------
    test_pred: DataFrame
        dataframe with labels, predictions, and errors
    path: str
        directory where plot should be saved.

    Returns
    -------
    None
    """
    # Longer code for numpy as pandas versions compatibility
    error_sums= test_pred.groupby('y').err.sum()
    plt.bar(error_sums.index.values, error_sums.values, width= 0.5)
    plt.xticks(error_sums.index.values)
    plt.ylabel('Target points')
    plt.xlabel('Predictions')
    plt.title('Sum of errors per points')
    plt.savefig(path + 'errors_per_point.png')
    plt.close()

def plot_preds_per_point(test_pred, path):
    """Plots mean prediction per label
    
    Parameters
    ----------
    test_pred: DataFrame
        dataframe with labels, predictions, and errors
    path: str
        directory where plot should be saved

    Returns
    -------
    None
    """
    # Longer code for numpy as pandas versions compatibility
    preds_mean= test_pred.groupby('y').pred.mean()
    plt.bar(preds_mean.index.values, preds_mean.values, width= 0.5)
    plt.xticks(preds_mean.index.values)
    plt.ylabel('Target points')
    plt.xlabel('Predictions')
    plt.title('Average prediction per target point')
    plt.ylim(80, 100)
    plt.savefig(path + 'preds_per_point.png')
    plt.close()

def spreadsheet_worst_preds(test_pred, test_x, path):
    """Compares the worst over and underestimations with instances features
    
    Parameters
    ----------
    test_pred: DataFrame
        dataframe with labels, predictions, and errors
    test_x: DataFrame
        test feature dataset
    path: str
        directory where plot should be saved.

    Returns
    -------
    None
    """   
    sorted_errs = test_pred.reset_index(drop=True).sort_values('err')
    data = pd.merge(sorted_errs, test_x, how='left', \
        right_index=True, left_index=True)

    data[:10].to_csv(path + 'overestimated.csv', index=False)
    data[-10:].to_csv(path + 'underestimated.csv', index=False)
