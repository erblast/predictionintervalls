



import sys
sys.path.append('./predintervals')

# import __init__ as m
import boot as b
# import __init__ as pi

if __name__ == '__main__':

    import doctest
    doctest.testmod( m = b, verbose=True
                    , optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                    )
    #doctest.testmod( m = pi, verbose=True, optionflags=doctest.ELLIPSIS)
