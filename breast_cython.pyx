import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#
# The arrays f, g and h is typed as "np.ndarray" instances. The only effect
# this has is to a) insert checks that the function arguments really are
# NumPy arrays, and b) make some attribute access like f.shape[0] much
# more efficient. (In this example this doesn't matter though.)


@cython.boundscheck(False) # turn off bounds-checking for entire function
def cy_search_prev(np.ndarray[dtype=np.int16_t, ndim=1] l_y, np.ndarray[dtype=np.int16_t, ndim=1] l_x, int y, int x, int count):
    
    cdef bint recently_found = False
    cdef int lim
    cdef int jj
    
    if(count < 20):
        lim = len(l_y) + 1
    else:
        lim = 20
        
    for jj in range(count,count-lim,-1):
        if(l_y[jj] == y) & (l_x[jj] == x):
            recently_found = True
            break
        
    return recently_found

