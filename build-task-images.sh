#!/usr/bin/env bash
if test -z "$1"
then
      echo "Usage ./build-task-images.sh VERSION"
      echo "No version was passed! Please pass a version to the script e.g. 0.1"
      exit 1
fi

VERSION=$1

docker build -t winery/download-data:$VERSION download_data

docker build -t winery/basic-image:latest base_images/basic

docker build -t winery/make-dataset:$VERSION make_dataset
docker build -t winery/process-dataset:$VERSION process_dataset

docker build -t winery/training-image:latest base_images/training

docker build -t winery/train-model:$VERSION train_model
docker build -t winery/evaluate-model:$VERSION evaluate_model
docker build -t winery/predict-model:$VERSION predict_model
