import click
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.pipeline import Pipeline
from transformer import Preprocessor


@click.command()
@click.option('--in-dir-processed')
@click.option('--in-dir-raw')
@click.option('--out-dir')
def train(in_dir_processed, in_dir_raw, out_dir):
    """Trains one XGBoost model for evaluation on test dataset
    and another one for deployment on full dataset

    Parameters
    ----------
    in_dir: Path
        directory where processed datasets are saved.
    out_dir: Path
        directory where trained model should be saved to.

    Returns
    -------
    None
    """   
    train_x = pd.read_csv(str(Path(in_dir_processed).parent / 'train_x.csv'), \
        index_col=0)
    train_y = pd.read_csv(str(Path(in_dir_processed).parent / 'train_y.csv'), \
        header= None, index_col=0)
    
    dmatrix = xgb.DMatrix(train_x, train_y.values)
    # best parameters from cross validation
    parameters = {
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror',
        'colsample_bytree': 0.6097,
        'gamma': 0.2582,
        'learning_rate': 0.0707,
        'max_depth': 4,
        'min_child_weight': 2,
        'reg_alpha': 0.5829,
        'reg_lambda': 0.7653,
        'subsample': 0.9154
        }

    xgb_model = xgb.train(parameters, dmatrix, num_boost_round=220)

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    xgb_model.save_model(out_dir + 'xgb_model.model')

    # xgb model to be trained on whole dataset and deployed
    full_dataset = pd.read_csv(str(Path(in_dir_raw) / \
        'wine_dataset.csv'), index_col=0)

    xgb_model_deploy = xgb.XGBRegressor(**parameters)
    xgb_pipeline = Pipeline([
        ('preprocessor', Preprocessor(full_dataset)),
        ('xgb_model', xgb_model_deploy)
        ])
    full_dataset_x = full_dataset.copy().drop('points', axis=1)
    full_dataset_y = full_dataset.points
    xgb_pipeline.fit(full_dataset_x, full_dataset_y)

    joblib.dump(xgb_pipeline, out_dir + 'xgb_pipeline.dat')

    flag = outdir / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    train()

