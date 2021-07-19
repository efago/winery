import click
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.pipeline import Pipeline
from transformer import Preprocessor


@click.command()
@click.option('--saved-model')
@click.option('--in-dir')
@click.option('--out-dir')
def predict_data(saved_model, in_dir, out_dir):
    """Predicts on new wine data

    Parameters
    ----------
    saved_model: str
        path where the trained model is saved as file.
    inp_dir: Path
        path of new data to be predicted.
    out_dir: Path
        directory where predictions should be saved to.

    Returns
    -------
    None
    """
    new_data = pd.read_csv(in_dir, index_col=0)
    pickled_model = joblib.load(Path(saved_model) / 'xgb_pipeline.dat')
    predictions = pickled_model.predict(new_data)

    new_data['Predicted point'] = np.round(predictions).astype('int')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_data.to_csv(str(out_dir / 'predictions.csv'))


if __name__ == '__main__':
    predict_data()