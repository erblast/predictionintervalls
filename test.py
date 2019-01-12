
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import predintervals as pi

# fit linear regression-----------------------------

boston = sklearn.datasets.load_boston()
boston.keys()
X = boston.data
y = boston.target
reg = LinearRegression()
reg.fit(X,y)
pred = reg.predict(X)


# regular example -----------------------------------

predintervals = pi.PredIntervals()
predintervals.fit( y, pred )
predintervals.predict( y, pred )

# dataframe example -----------------------------------

df = pd.DataFrame( dict(obs = y, pred = pred) ) \
    .reset_index()
    
predintervals = pi.PredIntervals()
predintervals.fit( df['obs'], df['pred'] )
predintervals.predict( df['obs'], df['pred'], df['index'] )

# feather example

df_trans = pd.read_feather('df_trans.feather') \
    .reset_index()
    
df_train = df_trans.query( 'data_set == "valid" & percentage == "10percent" & is_simulated == "no"') \
    .loc[:,['id', 'prediction', 'n_ae']]

predintervals.fit( obs = df_train['n_ae'], pred = df_train['prediction'] )

df_trans_sample = df_trans.iloc[0:100,:]

df_ecdf_pred = predintervals.predict( obs = df_trans_sample['n_ae']
                                     , pred = df_trans_sample['prediction']
                                     , ids = df_trans_sample['index'])
