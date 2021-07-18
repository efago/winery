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

    @property
    def image(self):
        return f'winery/train-model:{VERSION}'

    def requires(self):
        return ProcessDatasets()

    @property
    def command(self):
        return [
            'python', 'train.py',
            '--inp-dir', self.input().path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / \
                'xgb_model.model')
        )


class EvaluateModel(DockerTask):

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
            '--saved-model', self.input().path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path= str(Path(self.out_dir) / '.SUCCESS')
        )