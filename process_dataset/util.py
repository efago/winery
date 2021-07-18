import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def tokenize(sentence):
    """Tokenize sentence into words.
    
    This function tries to match lowercase words.
    Omits capitalized words, punctuations, and digits.

    Parameters
    ----------
    sentence: str
        sentence to be tokenized

    Returns
    -------
    str: whitespace joined lowercase words in the sentence
    """ 
    words = re.findall(r'\b[a-z]+\b', sentence)
    return ' '.join( words)


def fill_prices(df, df_train):
    """Fill Nan price values in dataframe
    
    This function uses mean encoding to fill Nan prices.
    First looks to fill Nan with average price of associated
    winery. Upon unavailability of winery average province
    prices are used. This function doesn't guarantee filling
    Nan for all instances.

    Parameters
    ----------
    df: DataFrame
        dataframe where Nan prices are to be filled
    df_train: DataFrame
        dataframe where average prices are referenced from

    Returns
    -------
    DataFrame: dataframe with most if not all price columns filled
    """   
    df.loc[df.price.isnull(), 'price']= df[df.price.isnull()] \
        .winery.map(lambda x: df_train[df_train.winery==x].price.mean())
    if any(df.price.isnull()):
        df.loc[df.price.isnull(), 'price']= df[df.price.isnull()].province \
        .map(lambda x: df_train[df_train.province==x].price.mean())

    return df


def get_vocab(df, sia) -> (set,set):
    """Retrieve discriminative words
    
    This function first looks at the descriptions in the
    top and bottom 3000 instances according to point assigned.
    Then selects the distinctive positive and negative words among 
    the groups.

    Parameters
    ----------
    df: DataFrame
        dataframe where Nan prices are to be filled
    df_train: DataFrame
        dataframe where average prices are referenced from

    Returns
    -------
    set: high and low sentiment words
    """       
    top_3000 = df.sort_values('points', ascending= False)[:3000]
    top_words = set()
    bottom_3000 = df.sort_values('points')[:3000]
    bottom_words = set()
    
    for desc in top_3000.description.map(tokenize):
        for word in desc.split(' '):
            if sia.polarity_scores(word)['compound'] > 0.3:
                top_words.add(word)
                
    for desc in bottom_3000.description.map(tokenize):
        for word in desc.split(' '):
            if sia.polarity_scores(word)['compound'] < 0:
                bottom_words.add(word)
    
    return top_words, bottom_words


def get_year(sentence):
    """Extract year from sentence
    
    Parameters
    ----------
    sentence: str
        string from where to extract year

    Returns
    -------
    int: a year between 1980 to 2019 if found, otherwise 0
    """ 
    year = re.findall(r'19[8|9][0-9]|20[0|1][0-9]', sentence)
    return int(year[0]) if len(year) > 0 else 0


def get_age(df, df_train):
    """Calculate relative age from the latest year
    
    Parameters
    ----------
    df: DataFrame
        dataframe where ages will be encoded
    df_train: DataFrame
        dataframe from where latest and average year is to
        be inferred

    Returns
    -------
    DataFrame: dataframe with age column
    """   
    year = df.title.map(get_year)
    if 'year' in df_train.columns:
        max_year = df_train.year.mean()
        mean_year = df_train.year.max()
    else:
        year_train = df_train.title.map(get_year)
        mean_year = year_train[ year_train > 0].mean()
        max_year = year_train.max()
    year[year == 0]= int(mean_year)
    df['age'] = year.map(lambda x: max_year - x)
    return df

def add_geography(df, df_train):
    """Assign numeric value representing geographic columns

    This function uses mean encoding to compute representative
    value from columns pertaining location with decreasing
    granularity of region_2, region_1, province, country.
    
    Parameters
    ----------
    df: DataFrame
        dataframe where geographic values will be encoded
    df_train: DataFrame
        dataframe from where average values will be inferred

    Returns
    -------
    DataFrame: dataframe with geographic column
    """
    region2_points = df_train.groupby('region_2').points.mean()
    df['_geography'] = df.region_2.map(region2_points)
    columns = ['country','province','region_1']

    while any(df._geography.isnull()):
        column = columns.pop()
        points = df_train.groupby(column).points.mean()
        df.loc[df._geography.isnull(), '_geography'] = \
            df[df._geography.isnull()][column].map(points)
        if len(columns) == 0:
            df.loc[df._geography.isnull(), '_geography'] = \
                df_train.points.mean()
    return df

def feature_engineer(df, df_train):
    """Feature engineer features of a dataframe

    This function computes numeric features for model training.
    
    Parameters
    ----------
    df: DataFrame
        dataframe where feature engineering is to be applied
    df_train: DataFrame
        dataframe from where different feature values will be inferred

    Returns
    -------
    DataFrame: dataframe with feature engineered columns
    """
    # column representing length of description
    df['desc_len']= df.description.str.len()
    
    df = fill_prices(df, df_train)
    df = add_geography(df, df_train)
    df = get_age(df, df_train)
    
    # column representing sentiment of description
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df.description.map( 
        lambda x: sia.polarity_scores(x)['compound'])
    
    top_words, bottom_words = get_vocab(df_train, sia)
    topv = CountVectorizer(vocabulary=top_words)
    bottomv = CountVectorizer(vocabulary=bottom_words)
    # columns representing total occurence of positive or negative words
    df['pos_words'] = topv.transform(df.description).toarray().sum(1)
    df['neg_words' ]= bottomv.transform(df.description).toarray().sum(1)
    
    df['_price'] = df.price.map(np.log)
    
    df.fillna(df_train.drop('points', axis=1).mean(axis= 0), inplace= True)
    df.fillna( df.mean(), inplace= True)
    
    return df