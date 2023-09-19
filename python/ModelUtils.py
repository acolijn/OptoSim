import numpy as np

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
        height, width = heatmap.shape

        # Reshape the heatmap to a 4D tensor for pooling
        heatmap_4d = heatmap.reshape((1, 1, height, width))

        # Calculate the scale factors for downsampling
        scale_factor_height = height // new_height
        scale_factor_width = width // new_width

        # Perform average pooling using np.mean
        downsampled_heatmap_4d = np.mean(heatmap_4d.reshape((1, 1, new_height, scale_factor_height, new_width, scale_factor_width)), axis=(3, 5))

        # Reshape the downsampled heatmap to the desired dimensions
        downsampled_heatmap = downsampled_heatmap_4d.reshape((new_height, new_width))

        downsampled_heatmaps.append(downsampled_heatmap)

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