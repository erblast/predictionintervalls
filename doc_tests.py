



import sys
sys.path.append('./predintervals')

import boot as b_file
import pi as pi_file

import predintervals as pi

import doctest

if __name__ == '__main__':
    

    doctest.testmod( m = b_file, verbose=True
                    , optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                    )
    doctest.testmod( m = pi_file, verbose=True, optionflags=doctest.ELLIPSIS)
    
    
    
    
    
