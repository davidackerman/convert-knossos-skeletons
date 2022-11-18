from scipy.optimize import curve_fit
import numpy as np

# np.s_[2::2]
# s definition: https://static-content.springer.com/esm/art%3A10.1186%2F1751-0473-9-16/MediaObjects/13029_2013_116_MOESM1_ESM.pdf

# NOTE: needs to be self avoiding?
# def fit_cos_theta(l, mean_cos_theta, stop_nm, start_nm=0, s=1):
#     valid_ids = np.where((l >= start_nm) & (l <= stop_nm))[0]
#     start_id = valid_ids[0]
#     stop_id = valid_ids[-1] + 1

#     def func(l, P):
#         return np.exp(-l / (s * P))

#     popt, pcov = curve_fit(func, l[start_id:stop_id], mean_cos_theta[start_id:stop_id])
#     return popt[0]


# def fit_R_squared(l, mean_R_squared, stop_nm, start_nm=0, s=1):
#     valid_ids = np.where((l >= start_nm) & (l <= stop_nm))[0]
#     start_id = valid_ids[0]
#     stop_id = valid_ids[-1] + 1

#     def func(l, P):
#         return 2 * s * P * l * (1 - s * (P / l) * (1 - np.exp(-l / (s * P))))

#     popt, pcov = curve_fit(func, l[start_id:stop_id], mean_R_squared[start_id:stop_id])
#     return popt[0]


# def fit_delta_squared(L, mean_delta_squared, stop_nm, start_nm=0, s=1):
#     valid_ids = np.where((L >= start_nm) & (L <= stop_nm))[0]
#     start_id = valid_ids[0]
#     stop_id = valid_ids[-1] + 1

#     # NOTE: l~L for L<<P
#     def func(L, P):
#         return (L**3) / (24 * s * P)

#     popt, pcov = curve_fit(
#         func, L[start_id:stop_id], mean_delta_squared[start_id:stop_id]
#     )
#     return popt[0]


def delta_squared_equation(L, P, s=1):
    return (L**3) / (24 * s * P)


def R_squared_equation(l, P, s=1):
    return 2 * s * P * l * (1 - s * (P / l) * (1 - np.exp(-l / (s * P))))


def cos_theta_equation(l, P, s=1):
    return np.exp(-l / (s * P))


def fit_func(func, dic, stop_nm, start_nm=0, s=1):
    x = np.array(list(dic.keys()))
    y = np.array(list(dic.values()))
    valid_ids = (x >= start_nm) & (x <= stop_nm)

    x = x[valid_ids]
    y = y[valid_ids]
    if len(x) > 0:
        popt, _ = curve_fit(lambda l, P: func(l, P, s), x, y)
        return popt[0]
    else:
        return np.NaN
