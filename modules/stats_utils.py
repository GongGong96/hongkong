"""
stats_utils.py
Implementasi ringan (tanpa scipy) untuk uji Chi-Square Goodness of Fit.

scipy==1.13.1 membutuhkan compiler Fortran (gfortran) untuk build dari
source, yang tidak tersedia di lingkungan deploy. Karena satu-satunya
fungsi scipy yang dipakai di aplikasi ini adalah `scipy.stats.chisquare`,
kita implementasikan ulang fungsi tersebut menggunakan `math` (stdlib)
dan `numpy` saja, sehingga dependency scipy bisa dihapus sepenuhnya.

Referensi algoritma incomplete gamma function: Numerical Recipes (gser/gcf).
"""

import math
import numpy as np

_ITMAX = 200
_EPS = 3e-16
_FPMIN = 1e-300


def _gser(a: float, x: float):
    """Deret (series) untuk regularized lower incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * _EPS:
            break
    gln = math.lgamma(a)
    return summ * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float):
    """Continued fraction untuk regularized upper incomplete gamma Q(a, x)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    gln = math.lgamma(a)
    return math.exp(-x + a * math.log(x) - gln) * h


def gammaq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("Argumen tidak valid untuk gammaq")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) dari distribusi chi-square, yaitu p-value."""
    if x <= 0:
        return 1.0
    return gammaq(df / 2.0, x / 2.0)


def chisquare(f_obs, f_exp):
    """
    Pengganti ringan untuk scipy.stats.chisquare(f_obs, f_exp).
    Mengembalikan tuple (chi2_statistic, p_value), sama seperti scipy.
    """
    f_obs = np.asarray(f_obs, dtype=float)
    f_exp = np.asarray(f_exp, dtype=float)

    chi2_stat = float(np.sum((f_obs - f_exp) ** 2 / f_exp))
    df = len(f_obs) - 1
    p_value = chi2_sf(chi2_stat, df)

    return chi2_stat, p_value
