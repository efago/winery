import click
import numpy as np
import pandas as pd
from pathlib import Path


def _save_datasets(train, test, outdir: Path):
    """Saves the train and test splitted data to local disk.

    Parameters
    ----------
    train: DataFrame
        data for model training.
    test: DataFrame
        data for model evaluation.
    out_dir:
        directory where file should be saved to.

    Returns
    -------
    None
    """
    out_train = outdir / 'train.csv/'
    out_test = outdir / 'test.csv/'
    flag = outdir / '.SUCCESS'

    train.to_csv(str(out_train))
    test.to_csv(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    """Split the raw data into training and test datasets

    Parameters
    ----------
    in_csv: str
        name of the raw csv file on local disk.
    out_dir: Path
        directory where files should be saved to.

    Returns
    -------
    None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddf = pd.read_csv(in_csv, index_col= 0)
    ddf = ddf.drop_duplicates(keep= 'last')
    ddf = ddf.dropna(axis= 0, how= 'all')
    n_samples = len(ddf)
    
    idx = np.random.choice(ddf.index, n_samples, replace= False)
    test_idx = idx[:n_samples // 10]
    test = ddf.loc[test_idx]

    train_idx = idx[n_samples // 10:]
    train = ddf.loc[train_idx]

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
