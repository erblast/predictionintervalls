import sklearn
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LinearRegression
import predintervals as pi
from tqdm import tqdm

class PredIntervals():
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
    >>> predintervals = pi.PredIntervals()
    >>> df_fit = predintervals.fit( y, pred )
    >>> df_fit.columns.tolist()
    ['cut', 'pred_min', 'pred_max', 'boot']
    >>> df = predintervals.predict( y, pred )
    >>> df.columns.tolist()
    ['id', 'cut', 'boot_stat', 'pred', 'obs', 'ecdf_results', 'me', 'sd', 'qu_025', 'qu_125', 'qu_5', 'qu_875']
    """
    
    def fit( self, obs, pred
             , step_size = 25
             , r = 1000
             , quantiles = [0.025, 0.125, 0.5, 0.875, 0.975]):
             
        assert len(obs) == len(pred), 'length of obs and pred does not match'
        
        df = pd.DataFrame( dict( obs = obs
                           , pred = pred
                           , id = id ) )  \
            .sort_values('pred') \
            .assign( rwn = range(0,len(pred) ) ) \
            .assign( cut = lambda x: pd.cut( x.rwn, len(pred)/step_size ) ) \
            .assign( cut = lambda x: x.cut.apply(str) ) \
            .assign( pred_min = lambda x : x.groupby('cut').transform('min')['pred'] ) \
            .assign( pred_max = lambda x : x.groupby('cut').transform('max')['pred'] ) \
            .groupby( ['cut', 'pred_min', 'pred_max'] ) \
            .aggregate( dict( obs = lambda x: tuple(x) ) ) \
            .reset_index()
            
        # we stretch this part so we can use the progressbar
        boots = []
        
        for cut in tqdm( df.cut.unique() ):
        
            df_slice = df.query('cut == "{}"'.format(cut) )
        
            boot = pi.Boot()
        
            boot.fit( df_slice['obs'].values[0] )
        
            boots.append(boot)
        
        
        df = df.assign( boot = boots )
        
        self.fitted = df.loc[:,['cut', 'pred_min', 'pred_max', 'boot']]
        
        return self.fitted
        
    def predict( self, obs, pred, ids = None):
        
        assert len(obs) == len(pred)
        
        # check id input
        if ids is None:
            ids = range(0, len(obs) )
        else:
            try:
                ids = ids.values
            except AttributeError as e:
                pass
                
            assert len(obs) == len(ids), 'sample and ids input have different lengths'
        
        df = pd.DataFrame( dict( obs = obs, pred = pred, id = ids) )
        
        # if dtype == int it will be converted to float for some reason
        # so we capture dtype here to convert back later
        id_dtype = df['id'].dtype
        
        # bring min and max prediction to training level
        
        pred_min_fit = self.fitted['pred_min'].min()
        pred_max_fit = self.fitted['pred_max'].max()
        
        def adjust_extremes_to_fitted(x):
            
            if x > pred_max_fit: x = pred_max_fit
            if x < pred_min_fit: x = pred_min_fit
            
            return x
            
        df['pred'] = df['pred'].apply(adjust_extremes_to_fitted)
        
        # merge with self.fitted which contains the segments(cut) and a Boot object
        # for each segment
        df_merge = pd.merge_asof(df.sort_values('pred')
                                 , self.fitted
                                 , left_on='pred'
                                 , right_on='pred_min') \
        
        # aggregate the observed values and the ids by segment/cut
        df_aggr = df_merge.groupby('cut') \
            .aggregate( dict( obs = lambda x: list(x)
                             , id = lambda x: list(x) ) ) \
            .reset_index()
        
        # rejoin wirh self.fitted to get boot objects. We had to discard them
        # because we cannot group on columns containing complex classes
        
        df_aggr = df_aggr \
            .merge( self.fitted, on = 'cut')
            
        boot_results = []
        
        for cut in tqdm( df_aggr['cut'].values ):
            
            boot_cut = df_aggr.loc[ df_aggr['cut'] == cut, 'boot'].values[0]
            obs_cut  = df_aggr.loc[ df_aggr['cut'] == cut, 'obs'].values[0]
            ids_cut  = df_aggr.loc[ df_aggr['cut'] == cut, 'id'].values[0]
            
            df_ecdf = boot_cut.predict(obs_cut, ids_cut)
            
            boot_results.append(df_ecdf)
            
        # the boot instances return a dataframe which can be concatenated into one
        df_concat = pd.concat( boot_results )
        
        # merge predictions and ids back in--------------------------------
        
        col = ['id','cut', 'boot_stat', 'pred', 'obs']
        col.extend( df_concat.columns.tolist()[3:-1] )
        
        df_final = df_concat.merge( df_merge.loc[:,['pred','id', 'cut']], on = 'id') \
            .rename( dict(values = 'obs'), axis = 1 ) \
            .loc[:,col] \
            .sort_values('id') \
            .reset_index(drop = True)
            
        # test correct id incorporation--------------------------------------
        assert df_final.shape[0] == len(obs) * 2
        # # debug code in case of assertion violation
        # print(df_final.shape[0])
        # print(len(obs) * 2)
        
        df_test = df_final.loc[:,['pred', 'obs', 'id']] \
            .assign( n = None ) \
            .groupby(['pred', 'obs','id']) \
            .aggregate( dict(n = 'size') ) \
            .reset_index() \
            .loc[:,['pred', 'obs','id']] \
            .sort_values('id') \
            .reset_index(drop = True) \
            .assign( id = lambda x : x['id'].astype(id_dtype) )
            
        df_id = df \
            .loc[:,['pred', 'obs','id']] \
            .sort_values('id') \
            .reset_index(drop = True)
            
        assert df_test.equals( df_id )
        # # debug in case of assertion violation
        # print( df_test.equals( df_id ))
        # print( df_test.head(10) )
        # print( df_id.head(10) )
        # print( df_test.tail(10) )
        # print( df_id.tail(10) )
        # print( df_test.shape )
        # print( df_id.shape )
        # return df_final, df_concat, df_merge, df_aggr
        
        return df_final

    
if __name__ == '__main__':
    
    boston = sklearn.datasets.load_boston()
    boston.keys()
    X = boston.data
    y = boston.target
    reg = LinearRegression()
    reg.fit(X,y)
    pred = reg.predict(X)
    
    predintervals = PredIntervals()
    predintervals.fit( y, pred )
    predintervals.predict( y, pred )
        
    # feather example

    # df_trans = pd.read_feather('df_trans.feather') \
    #     .reset_index()
    #
    # df_train = df_trans.query( 'data_set == "valid" & percentage == "10percent" & is_simulated == "no"') \
    #     .loc[:,['id', 'prediction', 'n_ae']]
    #
    # predintervals = PredIntervals()
    #
    #
    # predintervals.fit( obs = df_train['n_ae'], pred = df_train['prediction'] )
    #
    # predintervals.fitted
    #
    # # df_trans_sample = df_trans.iloc[0:100,:]
    # df_trans_sample = df_trans
    #
    # df_ecdf_pred = predintervals.predict( obs = df_trans_sample['n_ae']
    #                                      , pred = df_trans_sample['prediction']
    #                                      , ids = df_trans_sample['index'])
    # print(df_ecdf_pred)
    
