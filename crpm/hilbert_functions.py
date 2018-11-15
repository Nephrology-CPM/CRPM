"""Converting Hilbert Space filling curve index to Cartesian coordinates
"""

def hilbert_index2coord(order=1, index=0):
    """get 2D Cartesian coordinates from hilber cureve index
    """
    #initial value
    xcart = 0
    ycart = 0
    zindex = index
    slen = 1
    norder = 2**order

    while slen < norder:
        rxcart = ((zindex // 2) % 2)

        if rxcart == 0:
            rycart = (zindex % 2)
        else:
            rycart = ((zindex ^ rxcart) % 2)
        xcart, ycart = hilbert_rotatequadrant(slen, xcart, ycart, rxcart, rycart)
        xcart += slen*rxcart
        ycart += slen*rycart
        zindex = zindex//4
        slen = slen*2

    return(xcart, ycart)



def hilbert_rotatequadrant(slen, xcart, ycart, rxcart, rycart):
    """ rotates hilbert curve path by 90 degrees for each new block (quadrant)"""
    if rycart == 0:
        if rxcart == 1:
            xcart = slen-1-xcart
            ycart = slen-1-ycart
        xcart, ycart = ycart, xcart
    return(xcart, ycart)
