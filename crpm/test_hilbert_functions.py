""" test we can get map linear index to 2D coordinates
"""

def test_hilbert_index2coord():
    """ table of cases for N=4 mapping

    5-6 9-10
    | | | |
    4 7-8 11
    |     |
    3-2 *-12
      | |
    0-1 *-15

    """
    from crpm.hilbert_functions import hilbert_index2coord

    #index = 0 -> xy = (0, 0)
    assert hilbert_index2coord(2, 0) == (0, 0)

    #index = 2 -> xy = (1, 1)
    assert hilbert_index2coord(2, 2) == (1, 1)

    #index = 8 -> xy = (2, 2)
    assert hilbert_index2coord(2, 8) == (2, 2)

    #index = 10 -> xy = (3, 3)
    assert hilbert_index2coord(2, 10) == (3, 3)
