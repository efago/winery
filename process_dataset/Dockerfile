FROM winery/basic-image:latest

WORKDIR /opt/process_dataset
COPY . .

RUN pip install -r ./requirements.txt
RUN python -m nltk.downloader vader_lexicon