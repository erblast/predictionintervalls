
import sklearn
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    
boston = sklearn.datasets.load_boston()

boston.keys()

X = boston.data
y = boston.target
feats = boston.feature_names

reg = LinearRegression()

reg.fit(X,y)

pred = reg.predict(X)

df = pd.DataFrame( dict( obs = y
                   , pred = pred
                   , rwn = range(0,len(y) ) ) ) \
    .assign( cut = lambda x: pd.cut( x.rwn, len(y)/25 ) ) \
    .assign( cut = lambda x: x.cut.apply(str) )


for cut in df.cut.unique():
    
    cut = df.cut.unique()[0]
    
    slice = df.query('cut == "{}"'.format(cut) )
    
    
