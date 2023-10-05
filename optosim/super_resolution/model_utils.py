import numpy as np
import scipy.interpolate
import scipy.ndimage

def reshape_data(X):
    """
    Reshape the data to be compatible with the model.

    Returns:
        numpy.ndarray: The reshaped data.
    """
    # Get the dimensions of the data
    # Reshape the data
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL's congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [float]:
        a = np.cast[float](a)

    m1 = int(minusone)
    ofs = int(centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only supports "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                            * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[tuple(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                            * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + list(range(ndims - 1))
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = list(range(np.rank(newcoords)))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only 'neighbour', 'nearest', 'linear',",
              "and 'spline' are supported.")
        return None

def downsample_heatmaps_to_dimensions(heatmaps, new_height, new_width):
    """
    Downsample a list of heatmaps to specified dimensions using averaging.

    Args:
        heatmaps (list of numpy.ndarray): List of high-resolution heatmaps.
        new_height (int): The desired height of the downsampled heatmaps.
        new_width (int): The desired width of the downsampled heatmaps.

    Returns:
        list of numpy.ndarray: List of downsampled heatmaps.
    """
    downsampled_heatmaps = []

    for heatmap in heatmaps:
        # Get the dimensions of the original heatmap
        reshaped_heatmap = congrid(heatmap, (new_height, new_width), method='linear', centre=True, minusone=False)
        downsampled_heatmaps.append(reshaped_heatmap)

    return np.asarray(downsampled_heatmaps)



def weighted_average_estimator(X, r):
    """
    Estimate the position of the event by taking the weighted average of the PMTs.

    Args:
        X (numpy.ndarray): The data.

    Returns:
        list of tuple of float: The estimated positions.
    """
    x = (-r*X[:,0,0] + r*X[:,0,1] - r*X[:,1,0] + r*X[:,1,1])/np.sum(X, axis=(1,2))
    y =(-r*X[:,0,0] - r*X[:,0,1] + r*X[:,1,0]+ r*X[:,1,1])/np.sum(X, axis=(1,2))
    return list(zip(x,y))

def mse(true, pred):
    """
    Calculate the mean squared error between the true and predicted positions.

    Args:
        true (list of tuple of float): The true positions.
        pred (list of tuple of float): The predicted positions.

    Returns:
        float: The mean squared error.
    """
    return np.mean((np.asarray(true) - np.asarray(pred))**2)

def r_squared(true, pred):
    """
    Calculate the R^2 between the true and predicted positions.

    Args:
        true (list of tuple of float): The true positions.
        pred (list of tuple of float): The predicted positions.

    Returns:
        float: The R^2.
    """
    return 1 - np.sum((np.asarray(true) - np.asarray(pred))**2)/np.sum((np.asarray(true) - np.mean(true))**2)