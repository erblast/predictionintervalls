

from setuptools import setup

setup(name='predintervals',
      version='0.0.2',
      description='calculate prediction intervals for a given set of predicted and observed values',
      url='https://github.com/erblast/predintervals',
      author='Bjoern Koneswarakantha',
      author_email='bjoern.koneswarakantha@roche.com',
      license='GPL-3.0',
      packages=['predintervals'],
      install_requires = ['sklearn'
                          , 'pandas'
                          , 'numpy'
                          , 'scipy'
                          , 'resample'
                          , 'tqdm'
                          , 'statsmodels']
      , zip_safe=False)
