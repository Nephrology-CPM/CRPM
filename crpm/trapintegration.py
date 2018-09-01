"""implementation of trapezoidal integration"""

def trapintegration(curve):
    """Will add up trapezoids defined by points in x and y.

    Input:
        curve: list of real valued (x,y) pairs
    Result:
        The real valued integral.
    """
    cumval = 0.
    if len(curve) > 1:
        for idx in range(1, len(curve)):
            #print(idx,curve[idx],curve[idx][0],curve[idx][1])
            deltax = abs(curve[idx][0] - curve[idx-1][0])
            deltay = (curve[idx][1] + curve[idx - 1][1])
            cumval += deltax * deltay / 2.0

    return cumval
