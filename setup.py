

from setuptools import setup

setup(name='predictionintervalls',
      version='0.0.1',
      description='calculate prediction intervalls for a given set of predicted and observed values',
      url='https://github.com/erblast/predictionintervalls',
      author='Bjoern Koneswarakantha',
      author_email='bjoern.koneswarakantha@roche.com',
      license='GPL-3.0',
      packages=['predictionintervalls'],
      install_requires = ['sklearn'
                          , 'pandas'
                          , 'numpy'
                          , 'scipy'
                          , 'resample'
                          , 'tqdm'
                          , 'statsmodels']
      , zip_safe=False)
