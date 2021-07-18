import click
from pathlib import Path
import pandas as pd
import numpy as np
from util import feature_engineer


def _save_datasets(train_x, train_y, test_x, test_y, out_dir: Path):
    """Save the processed datasets to local disk.

    Parameters
    ----------
    train_x: DataFrame
        features of training data.
    train_y: Series
        labels of test data.
    test_x: DataFrame
        features of test data.
    test_y: Series
        labels of test data.
    out_dir: Path
        directory where files should be saved to.

    Returns
    -------
    None
    """    
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    out_train_x = outdir / 'train_x.csv/'
    out_train_y = outdir / 'train_y.csv/'
    out_test_x = outdir / 'test_x.csv/'
    out_test_y = outdir / 'test_y.csv/'
    flag = outdir / '.SUCCESS'

    train_x.to_csv(str(out_train_x))
    train_y.to_csv(str(out_train_y))
    test_x.to_csv(str(out_test_x))
    test_y.to_csv(str(out_test_y))

    flag.touch()


@click.command()
@click.option('--in-dir')
@click.option('--out-dir')
def prepare_datasets(in_dir, out_dir):
    """Feature engineer and transform datasets

    Parameters
    ----------
    in_dir: Path
        directory where train and test datasets are saved.
    out_dir: Path
        directory where files should be saved to.

    Returns
    -------
    None
    """
    train= pd.read_csv(str(Path(in_dir).parent / 'train.csv'))
    test= pd.read_csv(str(Path(in_dir).parent / 'test.csv'))

    train_xy = feature_engineer(train.copy(), train)
    test_y = test.points
    test.drop('points', axis= 1, inplace= True)
    test_x = feature_engineer(test, train_xy)
    
    features= ['age', 'desc_len', 'sentiment','pos_words', 'neg_words', \
        '_geography', '_price']

    _save_datasets(train_xy[ features], train_xy.points, 
                    test_x[ features], test_y, out_dir)


if __name__ == '__main__':
    prepare_datasets()


