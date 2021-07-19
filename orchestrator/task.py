import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '2')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'winery/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://raw.githubusercontent.com/efago/winery/main/dataset/wine_data/wine_dataset.csv'
    )

    @property
    def image(self):
        return f'winery/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/datasets/')

    @property
    def image(self):
        return f'winery/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        return [
            'python', 'dataset.py',
            '--in-csv', self.input().path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class ProcessDatasets(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/processed/')

    @property
    def image(self):
        return f'winery/process-dataset:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        return [
            'python', 'process.py',
            '--in-dir', self.input().path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModel(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/model/')
    in_dir_raw = luigi.Parameter(default='/usr/share/data/raw/')

    @property
    def image(self):
        return f'winery/train-model:{VERSION}'

    def requires(self):
        return ProcessDatasets()

    @property
    def command(self):
        return [
            'python', 'train.py',
            '--in-dir-processed', self.input().path,
            '--in-dir-raw', self.in_dir_raw,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path= str(Path(self.out_dir) / '.SUCCESS')
        )


class EvaluateModel(DockerTask):

    model_dir = luigi.Parameter(default='/usr/share/data/model/')
    out_dir = luigi.Parameter(default='/usr/share/data/evaluation/')

    @property
    def image(self):
        return f'winery/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return [
            'python', 'evaluate.py',
            '--saved-model', self.model_dir,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path= str(Path(self.out_dir) / '.SUCCESS')
        )


class PredictModel(DockerTask):

    model_dir = luigi.Parameter(default='/usr/share/data/model/')
    out_dir = luigi.Parameter(default='/usr/share/data/prediction/')
    in_data = luigi.Parameter(default='/usr/share/data/test/test.csv')

    @property
    def image(self):
        return f'winery/predict-model:{VERSION}'

    def requires(self):
        return EvaluateModel()

    @property
    def command(self):
        return [
            'python', 'predict.py',
            '--saved-model', self.model_dir,
            '--in-dir', self.in_data,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path= str(Path(self.out_dir) / 'predictions.csv')
        )