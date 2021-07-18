import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def similarities( df):
    
    tfv= TfidfVectorizer(min_df= 20, max_df= 0.5, stop_words= 'english', 
                    ngram_range= (1,2))
    sorted_points= df.sort_values('points')
    tfv_description= tfv.fit_transform( sorted_points.description
                                        .map( tokenize)).toarray()
    idxs= [i*20 for i in range(490)]
    similarity= cosine_similarity( tfv_description[ idxs])
    
    return pd.DataFrame( similarity, columns= sorted_points.points.iloc[idxs], 
                        index= sorted_points.points.iloc[idxs])
def yearify( sentence):
    year= re.findall( r'19[8|9][0-9]|20[0|1][0-9]', sentence)
    return int( year[0]) if len( year) > 0 else 0

def tokenize( sentence):
    words= re.findall( r'\b[a-z]+\b', sentence)
    return ' '.join( words)

def fill_prices( df, df_train):
    
    df.loc[ df.price.isnull(), 'price']= df[ df.price.isnull()]\
            .winery.map( lambda x: df_train[df_train.winery==x].price.mean())
    if any( df.price.isnull()):
        df.loc[ df.price.isnull(), 'price']= \
            df[ df.price.isnull()].province \
                .map( lambda x: df_train[df_train.province==x].price.mean())

    return df

def get_vocab( df, sia):
    
    top_3000= df.sort_values('points', ascending= False)[:3000]
    top_words= set()
    bottom_3000= df.sort_values('points')[:3000]
    bottom_words= set()
    
    for desc in top_3000.description.map( tokenize):
        for word in desc.split(' '):
            if sia.polarity_scores( word)['compound'] > 0.3:
                top_words.add( word)
                
    for desc in bottom_3000.description.map( tokenize):
        for word in desc.split(' '):
            if sia.polarity_scores( word)['compound'] < 0:
                bottom_words.add( word)
    
    return top_words, bottom_words

def get_age( df, df_train):
    
    year= df.title.map( yearify)
    if 'year' in df_train.columns:
        max_year= df_train.year.mean()
        mean_year= df_train.year.max()
    else:
        year_train= df_train.title.map( yearify)
        mean_year= year_train[ year_train > 0].mean()
        max_year= year_train.max()
    year[ year == 0]= int( mean_year)
    df['age']= year.map( lambda x: max_year - x)
    return df

def add_geography( df, df_train):

    region2_points= df_train.groupby('region_2').points.mean()
    df['_geography']= df.region_2.map( region2_points)
    columns= ['country','province','region_1']

    while any( df._geography.isnull()):
        column= columns.pop()
        points= df_train.groupby( column).points.mean()
        df.loc[ df._geography.isnull(), '_geography']= \
            df[ df._geography.isnull()][column].map( points)
        if len(columns) == 0:
            df.loc[ df._geography.isnull(), '_geography']= df_train.points.mean()
    return df

def feature_engineer( df, df_train):

    if df.equals( df_train):
        df['_winery']= df.groupby(['winery', 'variety']).points.transform( 'mean')
    else:
        winery_means= pd.DataFrame( df_train.groupby(['winery', 'variety']) \
                                       .points.mean()).reset_index()
        df= df.merge( winery_means, how= 'left', on= ['winery', 'variety'])
        df.rename( columns= {'points': '_winery'}, inplace= True)
        if any( df._winery.isnull()):
            df.loc[ df._winery.isnull(), '_winery']= \
                df[ df._winery.isnull()].winery \
                    .map( lambda x: df_train.loc[ df_train.winery == x] \
                         .points.mean())
    
    df[ 'desc_len']= df.description.map( len)
    
    df= fill_prices( df, df_train)
    df= add_geography( df, df_train)
    df= get_age( df, df_train)
    
    sia= SentimentIntensityAnalyzer()
    df['sentiment']= df.description.map( 
        lambda x: sia.polarity_scores(x)['compound'])
    
    top_words, bottom_words= get_vocab( df_train, sia)
    topv= CountVectorizer( vocabulary= top_words)
    bottomv= CountVectorizer( vocabulary= bottom_words)
    df['pos_words']= topv.transform( df.description).toarray().sum(1)
    df['neg_words']= bottomv.transform( df.description).toarray().sum(1)
    
    df['_price']= df.price.map( np.log)
    
    df.fillna( df_train.drop('points', axis= 1).mean( axis= 0), \
              inplace= True)
    df.fillna( df.mean(), inplace= True)
    
    return df