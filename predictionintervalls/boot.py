
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from resample import permutation, bootstrap, utils
from sklearn.datasets import load_boston
from statsmodels.distributions.empirical_distribution import ECDF

quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]

def sum_stats(x, fun, **kwargs):
    
    st = np.vstack( x.values )
    
    df = pd.DataFrame(st)
    
    agg = df.aggregate( lambda x: fun( x, **kwargs))
    
    return  [ agg.values ]
    
    
def get_stats(s):
    
    ecdf = ECDF(s)
    
    return {"mean": np.mean(s),
            "std": np.std(s, ddof=1),
            "ecdf": ecdf(x),
            "quantiles": pd.Series(s).quantile(quantiles).values
             }


def aggregate_boot_results(boot):
    
    df_res = pd.DataFrame(dict( boot = boot) ) \
        .assign( me = lambda x: x.iloc[:,0].apply( lambda x: x['mean'])
                , std = lambda x: x.iloc[:,0].apply( lambda x: x['std'])
                , ecdf = lambda x: x.iloc[:,0].apply( lambda x: x['ecdf'])
                , quantiles = lambda x: x.iloc[:,0].apply( lambda x: x['quantiles']) ) \
        .drop('boot', axis=1)
        
        
    df_agg = df_res\
        .aggregate( [lambda x: sum_stats(x, np.mean)
                     , lambda x: sum_stats(x, np.std, ddof=1) ] ) \
        .assign( agg = ('me', 'sd') ) \
        .set_index('agg')
        
    return df_agg
        

def shape(df_agg, x):
    
    results = list()
    
    for stat in ['me', 'sd']:
        
        df = pd.DataFrame( dict( values = x
                                     , ecdf = df_agg.loc[stat,'ecdf'][0]
                                     , sd = df_agg.loc[stat, 'std'][0][0]
                                     , me = df_agg.loc[stat, 'me'][0][0]
                                     , boot_stat = stat
                                     ) ) \
          .loc[:, ['boot_stat', 'values', 'ecdf', 'me', 'sd'] ]
                                     
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
    
def boot(x, r = 1000 , quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]):
    """
    calculates bootstrap statistics relevant for prediction intervalls of a
    given sample
    
    param x: array-like, sample
    param r: int, number of resamples, Default: 1000
    param quantiles: array-like, floats between 0-1 denoting the intervall
    boundaries that should be included.
    
    return: pandas DataFrame,
        boot_stat: either 'me' or 'se', so its either the mean or the sd aggregate
        values: the original values
        ecdf: result of statsmodels.distributions.empirical_distribution.ECDF
        me: mean result
        sd: sd result
        qu_*: quantile boundaries
        
    Example:
    
        
    """
    
    b = bootstrap.bootstrap(x, f=get_stats, b=r)
    
    df_agg = aggregate_boot_results(b)
    
    df =shape(df_agg, x)
    
    return df
    

if __name__ == '__main__':
    
    x = np.random.randn(25)

    print( boot(x) )

    
