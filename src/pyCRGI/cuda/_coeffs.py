# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np

from .._coeffs import GH


GH = np.array(GH, dtype = 'f8')


@cuda.jit('f8(i8,i8,i8,f8,f8,i8)', device = True)
def get_g_coeff(
    nn, mm,
    ll, tt, tc, nc,
):
    """
    Processes coefficients

    Args:
        nn : Index i
        mm : Index j
        ll : int
        tt : float
        tc : float
        nc : int
    Returns:
        g
    """

    if nn == 0 and mm == 0:
        return np.nan

    temp = ll

    if nn == 0 and mm == 1:
        temp -= 1

    if nn > 1:
        temp += nn ** 2 - 1

    if nn > 0:

        limit = nn + 1
        if limit >= mm:
            limit = mm

        if limit > 0:
            temp += 2 * limit - 1

    return tc * GH[temp] + tt * GH[temp + nc]


@cuda.jit('f8(i8,i8,i8,f8,f8,i8)', device = True)
def get_h_coeff(
    nn, mm,
    ll, tt, tc, nc,
):
    """
    Processes coefficients

    Args:
        nn : Index i
        mm : Index j
        ll : int
        tt : float
        tc : float
        nc : int
    Returns:
        h
    """

    if mm == 0:
        return np.nan

    temp = ll

    if nn > 0:
        temp += nn ** 2

    limit = nn + 1
    if limit > mm:
        limit = mm

    if limit > 0:
        temp += 2 * limit - 1

    return tc * GH[temp] + tt * GH[temp + nc]


@cuda.jit('Tuple([i8,i8,f8,f8,i8])(i8)', device = True)
def get_coeffs_prepare(year):
    """
    Prepares to processes coefficients

    Args:
        year : Between 1900.0 and 2030.0
    Returns:
        nmx, ll, tt, tc, nc
    """

    if year >= 2020.0:
        tt = year - 2020.0
        tc = 1.0
        # pointer for last coefficient in pen-ultimate set of MF coefficients...
        ll = 3060 + 195
        nmx = 13
        nc = nmx * (nmx + 2)
    else:
        tt = 0.2 * (year - 1900.0)
        ll = int(tt)
        tt = tt - ll
        # SH models before 1995.0 are only to degree 10
        if year < 1995.0:
            nmx = 10
            nc = nmx * (nmx + 2)
            ll = nc * ll
        else:
            nmx = 13
            nc = nmx * (nmx + 2)
            ll = int(0.2 * (year - 1995.0))
            # 19 is the number of SH models that extend to degree 10
            ll = 120 * 19 + nc * ll
        tc = 1.0 - tt

    return (
        nmx,  # int
        ll,  # int
        tt,  # float
        tc,  # float
        nc,  # int
    )
