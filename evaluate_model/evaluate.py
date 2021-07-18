import click
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from util import *


@click.command()
@click.option( '--saved-model')
@click.option( '--out-dir')
def generate_report( saved_model, out_dir):
    """Generates graphs and csv files for model evaluation

    Parameters
    ----------
    saved_model: str
        path where the trained model is saved as file.
    out_dir: Path
        directory where evaluations should be saved to.

    Returns
    -------
    None
    """
    data_path = '/usr/share/data/processed/'

    test_x = pd.read_csv(data_path + 'test_x.csv', index_col=0)
    test_y = pd.read_csv(data_path + 'test_y.csv', index_col=0,\
         header= None).values.squeeze()
    train_y = pd.read_csv(data_path + 'train_y.csv', index_col=0, \
        header= None).values.squeeze()
    train_mean = train_y.mean()

    xgb_model = xgb.Booster()
    xgb_model.load_model(saved_model)
    preds = xgb_model.predict(xgb.DMatrix(test_x))
    rmse = np.sqrt( np.mean( np.square(preds - test_y)))
    
    # dataframe of target values, predictions, and errors
    test_pred = pd.DataFrame({'y' : test_y, 'pred' : preds})
    test_pred['err_abs'] = abs(test_pred.y - test_pred.pred)
    test_pred['err'] = test_pred.y - test_pred.pred

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # indexes of sorted target point values to compare predictions
    idx= np.argsort( test_y)
    
    plot_feature_importance(xgb_model, out_dir)
    plot_regression(test_y, preds, idx, out_dir)
    plot_distribution(test_y, preds, out_dir)    
    plot_error_cumulative(test_y, preds, idx, rmse, out_dir)
    compare_model_baseline(test_y, preds, idx, train_mean, out_dir)
    plot_error_per_point(test_pred, out_dir)
    plot_preds_per_point(test_pred, out_dir)
    spreadsheet_worst_preds(test_pred, test_x, out_dir)

    flag= Path( out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    generate_report()