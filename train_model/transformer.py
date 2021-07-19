from util import feature_engineer

from sklearn.base import TransformerMixin, BaseEstimator

class Preprocessor(TransformerMixin, BaseEstimator):
    """Custom transformer class to incorporate preprocessing
    in model pipeline
    
    Performs all the preprocessing done in the previous ProcessDatasets
    task.

    Parameters
    ----------
    TransformerMixin: class
        Base mixin class for all transformers in scikit-learn
    BaseEstimator: class
        Base estimator class for all estimators in scikit-learn

    Returns
    -------
    DataFrame: processed dataframe
    """ 
    def __init__(self, train_df):
        self.train_df = train_df

    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        columns = ['age', 'desc_len', 'sentiment','pos_words', 'neg_words', '_geography', '_price']
        data_processed = feature_engineer(x, self.train_df)
        return data_processed[columns]