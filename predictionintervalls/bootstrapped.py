
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats

df = pd.DataFrame()
df['revenue'] = np.random.normal(loc=100, scale=90, size=1000000)
df['revenue'] = df['revenue'].apply(lambda x: x if x > 0 else 1)

df['clicks'] = np.random.binomial(100, 0.15, 1000000)

sample_df = df[:5000]


# Calculate the estimate of revenue per record
print(df.revenue.mean())

print(bs.bootstrap(values=sample_df.revenue.values, stat_func=bs_stats.mean))


def custom_mean(values, axis=1):
    '''Calculate the mean of values for each bootstrap sample
    Args:
        values: a np.array of values we want to calculate the statistic on
            This is actually a 2d array (matrix) of values. Each row represents
            a bootstrap resample simulation that we wish to aggretage across.
    '''
    
    
    mean_values = np.mean(np.asmatrix(values),axis=axis)
    sd_values = np.std(np.asmatrix(values),axis=axis)
    
    if values.shape[0] != 1:
        # this function gets called 2x
        # once for the bootstrap and once to calculate the bootstrap statistics
        
        # once to calculate the statistic on the whole population (non bootstrap)
        #  filter out this case
        
        # 12345 bootstrap resample simulations
        print('function input shape {}'.format(values.shape))
        print('function output shape {} == num bootstrap iterations'.format(mean_values.shape))
        
    
    return [ mean_values , sd_values, sd_values ]

print('length of the array %d' % len(sample_df.revenue.values))

results = bs.bootstrap(sample_df.revenue.values, stat_func=custom_mean, num_iterations=12345)
print('')
print(results)
