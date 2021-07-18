import click
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path


@click.command()
@click.option('--inp-dir')
@click.option('--out-dir')
def train(inp_dir, out_dir):
    """Trains an XGBoost model

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
    train_x = pd.read_csv(str(Path(inp_dir).parent / 'train_x.csv'), \
        index_col=0)
    train_y = pd.read_csv(str(Path(inp_dir).parent / 'train_y.csv'), \
        header= None, index_col=0)
    
    dmatrix = xgb.DMatrix(train_x, train_y.values)
    # best parameters from cross validation
    parameters = {'eval_metric': 'rmse',
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

if __name__ == '__main__':
    train()

