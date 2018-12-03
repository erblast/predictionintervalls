
import numpy as np
import scipy.stats as stats
import pandas as pd
from resample import permutation, bootstrap, utils
from sklearn.datasets import load_boston
from statsmodels.distributions.empirical_distribution import ECDF

class Boot():
    
    
    def __init__(self):
                
        pass
        
    def _sum_stats(self, x, fun, **kwargs):
        """
        param x: array-like or a Series with arrays (same length for each array)
        param fun: a window function
        **kwargs: additional arguments passed to fun
        
        _sum_stats applies a window function to an array or a Series containing arrays
        by either creating a dataframe with one column when given a single arrays
        or a a dataframe with one column for each element in the array when given a
        Series of same-length arrays.
        
        return: list
        """
        
        st = np.vstack( x.values )
        
        df = pd.DataFrame(st)
        
        agg = df.aggregate( lambda x: fun( x, **kwargs))
        
        if( len(agg.values) == 1 ):
            return agg.values[0]
        else:
            return  [ agg.values ]
        
        
    def _get_stats(self, sample_boot):
        """
        this function is passed to the boot function of the resample package
        """
        
        ecdf = ECDF(sample_boot)
            
        return {'me': np.mean(sample_boot),
                'sd': np.std(sample_boot, ddof=1),
                'ecdf_function': ecdf,
                'quantiles': pd.Series(sample_boot).quantile(self.quantiles_input).values
                 }


    def _aggregate_boot_results(self, boot_results):
        """
        extracts bootstraps results into dataframe and aggregates the results.
        updates the class attributes me, sd, quantiles and ecdfs
        """
        
        # extract bootstrap results into dataframe
        # dataframe will have one row per bootstrap iteration
        df_res = pd.DataFrame(dict( boot = boot_results) ) \
            .assign( me         = lambda x: x.iloc[:,0].apply( lambda x: x['me'])
                    , sd        = lambda x: x.iloc[:,0].apply( lambda x: x['sd'])
                    , ecdfs     = lambda x: x.iloc[:,0].apply( lambda x: x['ecdf_function'])
                    , quantiles = lambda x: x.iloc[:,0].apply( lambda x: x['quantiles'])
                    ) \
            .drop('boot', axis=1)
            
        # aggregate bootstrap iteration to mean and SE
        df_agg = df_res \
            .drop('ecdfs', axis = 1) \
            .aggregate( [lambda x: self._sum_stats(x, np.mean)
                         , lambda x: self._sum_stats(x, np.std, ddof=1) ] ) \
            .assign( agg = ('me', 'sd') ) \
            .set_index('agg')
            
        # construct quantiles df
                
        def name_quant_vals(x):
                    
            x = x[0]
            
            names_quantiles = [ ('qu_' + str(q)).replace('0.','') for q in self.quantiles_input ]
            
            assert len(names_quantiles) == len(x), "len quantiles {} and len input {} do not match".format(names_quantiles, x)
            
            named_quantiles_tup  = ( (name, val) for name, val in zip(names_quantiles, x ) )
            named_quantiles_dict = dict(named_quantiles_tup)
            named_quantiles_df   = pd.DataFrame(named_quantiles_dict, index = [1])
            
            return named_quantiles_df
        
        df_agg = df_agg.assign( quantiles = lambda x: x['quantiles'].apply(name_quant_vals) )
            
        self.me = df_agg['me']
        self.sd = df_agg['sd']
        self.ecdfs = df_res['ecdfs'].values
        self.quantiles = pd.concat( df_agg['quantiles'].values, axis = 0, ignore_index=True )
        self.quantiles.index = df_agg.index
                                   
            
        return None
            
        
        
    def fit( self, sample, r = 1000 , quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]):
        """
        calculates bootstrap statistics and fits an ecdf for each sample.
        
        param sample: array-like, sample
        param r: int, number of resamples, Default: 1000
        param quantiles: array-like, floats between 0-1 denoting the interval
        boundaries that should be included.
        
        updates the class attributes me, sd, quantiles and ecdfs
            
        Example:
        >>> x = np.random.seed(1)
        >>> x = np.random.randn(25)
        >>> boot = Boot()
        >>> boot.fit(x)
        >>> boot.me
        agg ...
        >>> boot.sd
        agg ...
        >>> boot.quantiles
        qu_025 ...
        >>> len( boot.ecdfs )
        1000
        >>> boot.ecdfs[0]
        <statsmodels.distributions.empirical_distribution.ECDF ...
        
        Tests:
        # take pd.Series
        >>> boot.fit( pd.Series(x) )
        """
        
        # convert pd.Series to values
        try:
            sample = sample.values
        except AttributeError as e:
            pass
        
        self.r = r
        self.quantiles_input = quantiles
        
        f = lambda sample_boot: self._get_stats(sample_boot)
        
        b = bootstrap.bootstrap( sample, f = f , b=r)
        
        self._aggregate_boot_results(b)
        
        return None
        
    def predict(self, sample, ids = None):
        """
        calculate ecd values for a set of values using the fitted ecdfs
        
        return: pandas DataFrame,
            id: id column, will add sequential id if none provided
            boot_stat: either 'me' or 'se', so its either the mean or the sd aggregate
            values: the original values
            ecdf: result of statsmodels.distributions.empirical_distribution.ECDF
            me: mean result
            sd: sd result
            qu_*: quantile boundaries
        
        Examples:
        >>> x = np.random.seed(1)
        >>> x = np.random.randn(25)
        >>> boot = Boot()
        >>> boot.fit(x)
        >>> df_pred = boot.predict(x)
        >>> df_pred.columns.tolist()
        ['id', 'boot_stat', 'values', 'ecdf_results', 'me', 'sd', 'qu_025', 'qu_125', 'qu_5', 'qu_875', 'qu_975']
        
        # IDs will be maintained
        >>> boot = Boot()
        >>> boot.fit(x)
        >>> ids = range(25, len(x) + 25 )
        >>> df_pred = boot.predict(x, ids)
        >>> df_pred['id'].sort_values().unique().tolist()[0:10]
        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        """
        
        # convert pd.Series to values
        try:
            sample = sample.values
        except AttributeError as e:
            pass


        # check id input
        if ids is None:
            ids = range(0, len(sample) )
        else:
            
            try:
                ids = ids.values
            except AttributeError as e:
                pass
                
            assert len(sample) == len(ids), 'sample and ids input have different lengths'
        
        apply_ecdfs = lambda x: [ f(x) for f in self.ecdfs ]
        
        df_id = pd.DataFrame( dict( values = sample, id = ids) )
        
        df = pd.DataFrame( dict( values = sample ) ) \
             .assign( ecdf_results = lambda x: x['values'].apply( apply_ecdfs ) ) \
             .assign( me = lambda x: x.ecdf_results.apply( np.mean )
                      , sd = lambda x: x.ecdf_results.apply( lambda x: np.std(x, ddof=1) ) ) \
             .drop('ecdf_results', axis = 1) \
             .set_index( 'values' ) \
             .stack() \
             .reset_index() \
             .rename( { 'level_1' : 'boot_stat', 0 : 'ecdf_results' }, axis = 1 ) \
             .set_index('boot_stat') \
             .merge( self.me.to_frame(), left_index = True, right_index = True) \
             .merge( self.sd.to_frame(), left_index = True, right_index = True) \
             .merge( self.quantiles, left_index = True, right_index = True) \
             .reset_index() \
             .rename( dict(index = 'boot_stat'), axis = 1) \
             .sort_values('values') \
             .reset_index(drop = True)
        
        # bring id back in
        col = ['id']
        col.extend( df.columns.tolist() )
        
        df = df.merge(df_id, on = 'values') \
            .drop_duplicates() \
            .loc[:,col]
        
        # test correct id incorporation
        assert df.shape[0] == len(sample) * 2
        
        df_test = df.loc[:,['values','id']] \
            .assign( n = None ) \
            .groupby(['values','id']) \
            .aggregate( dict(n = 'size') ) \
            .reset_index() \
            .loc[:,['values','id']] \
            .sort_values('id')\
            .reset_index(drop = True)
            
        df_id = df_id \
            .sort_values('id') \
            .reset_index(drop = True)
            
        assert df_test.equals( df_id )
            
        return df
             

if __name__ == '__main__':
    
    np.random.seed(1)
    x = np.random.randn(25)

    boot = Boot()

    boot.fit(x)
    
    print(boot.me)
    print(boot.sd)
    print(boot.quantiles)
    print(len( boot.ecdfs ) )
    print( boot.ecdfs[0] )
    
    print(boot.predict(x))
    
