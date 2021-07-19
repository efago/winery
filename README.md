# Winery data

The dataset consists of wine instances of different categories, countries, prices, titles, and were rated by different tasters. It consists of 12 features and a rating label. The columns of the dataset and their likely representations are:

* Country – country where wine was produced
* Province – province where wine was produced
* Description – description of wine by the taster
* Price – price of wine
* Region_1 – county or district where wine was produced
* Region_2 – city or town where wine was produced
* Winery – winery that produced the wine
* Taster_name – person who rated the wine
* Taster_twitter_handle – twitter account of the rater
* Title – name of the wine
* Designation – vineyard where wine was produced
* Variety – type of wine

This repository includes an end to end ML pipeline that includes data ingestion, preprocessing, feature engineering, model training, model evaluation, and deployment.

>It uses Luigi for dependency resolution, workflow management, and container orchestration. Once you have the Docker daemon running, build the images by executing:

    ./build-task-images.sh 1

>The 1 argument could be changed to represent the version of the images. After the images are built the containers could be span by executing:

docker-compose up orchestrator

This would output the points prediction in the data_root/predictions folder.

>The default setting could be changed to fetch data from a different source as well as to make predictions on a different test dataset. This could be accomplished by changing the command at the docker-compose.yml file:

    command: luigi --module task PredictModel --scheduler-host luigid

>could be modified as:

    command: luigi --module task PredictModel --DownloadData-url <url or path of data for building model> --PredictModel-in-data <url or path of data to be predicted> --scheduler-host luigid

>If the data is locally stored, it should be inside the data_root folder and could be accessed from the Docker storage mount directory. Example:

    /usr/share/data/********.csv
