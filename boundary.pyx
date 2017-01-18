import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
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

def trace_boundary( np.ndarray im ):

    cdef int y = np.where(im != 0)[0][0]
    cdef int x = np.where(im != 0)[1][0]
    
    print np.where(im != 0)[0]
    print y
    print np.where(im != 0)[1]
    print x

    cdef np.ndarray l_y = np.zeros([10000], dtype=np.uint16)
    cdef np.ndarray l_x = np.zeros([10000], dtype=np.uint16)
    
    cdef int count = 0
    
    cdef np.ndarray y_s = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int)
    cdef np.ndarray x_s = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int)
    
    cdef bint first = True
    cdef bint found = False
    cdef int ii, jj
    
    #will remove pixels at the border, as these will sometimes give weird errors
    im[0:10,:] = 0
    im[-1:-11,:] = 0
    im[:,0:10] = 0
    
    
    while((x >= 0) & (y >= 0) & (y < im.shape[0])):
        l_y[count] = y
        l_x[count] = x
        found = 0
        print y_s.size
        for ii in range(0,np.size(y_s)):
            #first check that the indicies we will be using are valid
            if((x + x_s[ii]) >= 0) & ((x + x_s[ii]) < im.shape[1]) & ((y + y_s[ii]) >= 0) & ((y + y_s[ii]) < im.shape[0]):
                #found[ii] = (im[y + y_s[ii], x + x_s[ii]] == 1)
                #if the pixel we have found is on the boundary
                print('test = %d' %im[y + y_s[ii], x + x_s[ii]])
                if(im[y + y_s[ii], x + x_s[ii]] == 1):
                    
                    #if it is the first pass through, we will add this point
                    if(first):
                        x += x_s[ii]
                        y += y_s[ii]
                        print('First')
                        first = False
                        found = True
                        break
                    
                    #otherwise check we havent already found this point
                    elif((l_x[-2] != x + x_s[ii]) | (l_y[-2] != y + y_s[ii])):
                        x = x + x_s[ii]
                        y = y + y_s[ii]
                        found = True
                        #print np.max(l_x)
                        break
                    
                    else:
                        pass
            #print x
            #print y
        if(not found):
            break
        
        count += 1
    print('count = %d' %count)
    #now re-write the breast boundary in the class
    
    return l_y[0:count], l_x[0:count]
