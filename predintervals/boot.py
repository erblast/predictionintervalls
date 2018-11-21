
import numpy as np
import scipy.stats as stats
import pandas as pd
from resample import permutation, bootstrap, utils
from sklearn.datasets import load_boston
from statsmodels.distributions.empirical_distribution import ECDF


def sum_stats(x, fun, **kwargs):
    
    st = np.vstack( x.values )
    
    df = pd.DataFrame(st)
    
    agg = df.aggregate( lambda x: fun( x, **kwargs))
    
    return  [ agg.values ]
    
    
def get_stats(sample_boot, sample_original, quantiles):
    
    ecdf = ECDF(sample_boot)
        
    return {'mean': np.mean(sample_boot),
            'std': np.std(sample_boot, ddof=1),
            'ecdf_results': ecdf(sample_original),
            'ecdf_function': ecdf
            'quantiles': pd.Series(sample_boot).quantile(quantiles).values
             }


def aggregate_boot_results(boot):
    
    df_res = pd.DataFrame(dict( boot = boot) ) \
        .assign( me = lambda x: x.iloc[:,0].apply( lambda x: x['mean'])
                , std = lambda x: x.iloc[:,0].apply( lambda x: x['std'])
                , ecdf = lambda x: x.iloc[:,0].apply( lambda x: x['ecdf_results'])
                , quantiles = lambda x: x.iloc[:,0].apply( lambda x: x['quantiles']) ) \
        .drop('boot', axis=1)
        
        
    df_agg = df_res\
        .aggregate( [lambda x: sum_stats(x, np.mean)
                     , lambda x: sum_stats(x, np.std, ddof=1) ] ) \
        .assign( agg = ('me', 'sd') ) \
        .set_index('agg')
        
    return df_agg
        

def shape(df_agg, x, quantiles):
    
    results = list()
    
    for stat in ['me', 'sd']:
        
        df = pd.DataFrame( dict( values = x
                                     , ecdf = df_agg.loc[stat,'ecdf_results'][0]
                                     , sd = df_agg.loc[stat, 'std'][0][0]
                                     , me = df_agg.loc[stat, 'me'][0][0]
                                     , boot_stat = stat
                                     ) ) \
          .loc[:, ['boot_stat', 'values', 'ecdf_results', 'me', 'sd'] ]
                                     
        names_quantiles = [ ('qu_' + str(q)).replace('0.','') for q in quantiles  ]
        tuple_quantiles = ( (name, val) for name, val in zip(names_quantiles, df_agg.loc[stat,'quantiles'][0])  )
        dict_quantiles = dict(tuple_quantiles)
        df_quantiles = pd.DataFrame(dict_quantiles, index=[1])
        
        df_quantiles = pd.concat( [df_quantiles] * len(x) ) \
            .reset_index() \
            .drop('index', axis = 1)
            
        df = pd.concat( [df, df_quantiles] , axis = 1 )
        
        results.append(df)
        
    return pd.concat( results, axis = 0, ignore_index=True )
    
def boot( sample, r = 1000 , quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]):
    """
    calculates bootstrap statistics relevant for prediction intervals of a
    given sample
    
    param sample: array-like, sample
    param r: int, number of resamples, Default: 1000
    param quantiles: array-like, floats between 0-1 denoting the interval
    boundaries that should be included.
    
    return: pandas DataFrame,
        boot_stat: either 'me' or 'se', so its either the mean or the sd aggregate
        values: the original values
        ecdf: result of statsmodels.distributions.empirical_distribution.ECDF
        me: mean result
        sd: sd result
        qu_*: quantile boundaries
        
    Example:
    >>> x = np.random.randn(25)
    >>> df_boot = boot(x)
    >>> df_boot.shape
    (50, 10)
    >>> df_boot.columns.format()
    ['boot_stat', 'values', 'ecdf_results', 'me', 'sd', 'qu_025', 'qu_125', 'qu_5', 'qu_875', 'qu_975']
    >>> df_boot['boot_stat'].unique().tolist()
    ['me', 'sd']
    
    Tests:
    # take pd.Series
    >>> df_boot = pd.Series( x )
    """
        
    try:
        sample = sample.values
    except AttributeError as e:
        pass
    
    f = lambda sample_boot: get_stats(sample_boot, sample_original = sample
                                      , quantiles = quantiles)
    
    b = bootstrap.bootstrap( sample, f = f , b=r)
    
    df_agg = aggregate_boot_results(b)
    
    df = shape(df_agg, sample, quantiles)
    
    return df
    

if __name__ == '__main__':
    
    quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]
    
    x = np.random.randn(25)
    
    df_boot = boot(x)
    
    print( df_boot.shape )
    
    print( df_boot.columns.format() )
    
    print( df_boot['boot_stat'].unique() )
    
    df_boot = pd.Series( x )

    
