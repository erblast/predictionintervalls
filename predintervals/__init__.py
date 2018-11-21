
import sklearn
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LinearRegression
import predintervals as pi
from predintervals.boot import boot
from tqdm import tqdm

def predintervals( obs, pred
                         , id = None
                         , step_size = 25
                         , r = 1000
                         , quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]):
    """
    for a given set of observations and predictions calculates prediction
    intervals, mean and sd and single ecdf values. Sorts the predictions
    and groups them into steps for each of which the summary statistics will
    be bootstrapped.
    
    param obs: array-like floats, observations matching pred
    param pred: array-like floats, predictions matching obs
    param ids: array-like, ids will be preserved in returned dataframe
    param step_size: int, number of predictions to group into one step
    param r: int, number of resamples
    param quantiles: array-like, floats between 0-1 denoting the interval
    boundaries that should be included.
    
    return: pandas DataFrame,
        boot_stat: either 'me' or 'se', so its either the mean or the sd aggregate
        values: the original values
        ecdf: result of statsmodels.distributions.empirical_distribution.ECDF
        me: mean result
        sd: sd result
        qu_*: quantile boundaries
        
    Examples:
    >>> boston = sklearn.datasets.load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> reg = LinearRegression()
    >>> reg.fit(X,y)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> pred = reg.predict(X)
    
    >>> df_pi = pi.predintervals(pred = pred, obs = y)
    >>> df_pi.shape
    (1012, 14)
    >>> df_pi.columns.format()
    ['id', 'obs', 'pred', 'rwn', 'cut', 'boot_stat', 'ecdf_results', 'me', 'sd', 'qu_025', 'qu_125', 'qu_5', 'qu_875', 'qu_975']
    >>> df_pi['boot_stat'].unique().tolist()
    ['me', 'sd']
    """
    
    assert len(obs) == len(pred), 'length of obs and pred does not match'
    
    if id is None:
        id = range(0, len(obs) )
    
    df = pd.DataFrame( dict( obs = obs
                       , pred = pred
                       , id = id ) )  \
        .sort_values('pred') \
        .assign( rwn = range(0,len(pred) ) ) \
        .assign( cut = lambda x: pd.cut( x.rwn, len(pred)/step_size ) ) \
        .assign( cut = lambda x: x.cut.apply(str) )

    results = []
    
    for cut in tqdm( df.cut.unique() ):
        
        df_slice = df.query('cut == "{}"'.format(cut) )

        df_boot = boot( df_slice['obs'].values, r, quantiles )

        df_tot = pd.concat( [df_slice] * 2, axis = 0, ignore_index=True ) \
            .reset_index() \
            .drop('index', axis = 1)

        df_tot = pd.concat( [df_tot, df_boot], axis = 1 ) \
            .query( 'obs == values') \
            .drop('values', axis = 1)
        
        assert df_tot.shape[0] == ( df_slice.shape[0] * 2 )

        results.append(df_tot)

    df_results = pd.concat( results
                           , axis = 0
                           , ignore_index=True )
    
    
    return df_results.sort_values('id')
    
if __name__ == '__main__':
    
    boston = sklearn.datasets.load_boston()

    boston.keys()

    X = boston.data
    y = boston.target

    reg = LinearRegression()

    reg.fit(X,y)

    pred = reg.predict(X)

    df_pi = predintervals(pred = pred, obs = y)
            
    print( df_pi.shape)
    
    print( df_pi.columns.format() )
    
