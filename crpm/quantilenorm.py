""" quantile normalize matricies by rows """

def quantilenorm(matrixlist):
    """ normalizes matricies to have corresponding rows with similar distributions
    Args:
        matrixlist: a list of 2D array-like objects all with same number of rows.

    Returns:
        A matrix list of the transformed original matricies.
    """

    # - check input -
    #assert input is a list
    if not isinstance(matrixlist, list):
        print("Error quantilenorm: input is not a list")
        return None
    #assert elements of list are arraylike
    for elem in 



    return matrixlist
