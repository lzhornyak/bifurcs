# import PyDSTool as dst
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations
from string import ascii_lowercase
import sympy as sp
from tqdm import tqdm
from mpire import WorkerPool
from functools import partial

from matplotlib.contour import QuadContourSet
from scipy import ndimage
import contourpy
from scipy.signal import find_peaks
from scipy.special import comb


def generate_equations(bases, params, append=None):
    var_sets = [s for i in range(1, len(bases) + 1) for s in combinations(bases, i)]
    equations = []
    for v in var_sets:
        equation = ' + '.join(v)
        if append is not None:
            equation = equation + ' + ' + append
        equations.append((equation, [p for p in params if p in equation]))
    return equations


def sample_points(equation, params, system_var='x', bifurc_var='r', param_values=None, abs_params='', mesh_size=100):
    r, x = sp.symbols(bifurc_var), sp.symbols(system_var)
    params = sp.symbols(params)
    abs_params = sp.symbols(abs_params) if abs_params else []
    equation = sp.sympify(equation)
    if param_values is None:
        param_values = np.random.rand(len(params)) * 2 - 1
    for p, pv in zip(params, param_values):
        if p in abs_params: pv = abs(pv)
        equation = equation.subs(p, pv)

    f = sp.lambdify((r, x), equation, 'numpy')
    points = np.meshgrid(np.linspace(-1, 1, mesh_size), np.linspace(-5, 5, mesh_size))
    return equation, (points[0], points[1], f(points[0], points[1]))


def split_lines(sequence, deadzone=0):
    # split sequence into monotonic segments
    smoothed = ndimage.gaussian_filter1d(sequence.copy().T, 3)
    second_diff = np.abs(np.diff(np.diff(smoothed)))
    try:
        peaks_r = find_peaks(second_diff[0], prominence=second_diff[0].max() / 2)[0]
    except ValueError:
        peaks_r = []
    try:
        peaks_x = find_peaks(second_diff[1], prominence=second_diff[1].max() / 2)[0]
    except ValueError:
        peaks_x = []
    splits = np.concatenate([np.array([0]), peaks_x, peaks_r, np.array([len(sequence)])])
    splits.sort()
    split_sequence = []
    for sl, sr in zip(splits[:-1], splits[1:]):
        try:
            new_seq = sequence[sl + deadzone:sr - deadzone]
            if len(new_seq) < 20:
                continue
            if new_seq[-1, 0] - new_seq[0, 0] < 0:  # flip if decreasing
                new_seq = new_seq[::-1]
        except TypeError:
            continue
        split_sequence.append(new_seq)
    return split_sequence


def resample_line(sequence, n_samples=100):
    r = np.linspace(sequence[0, 0], sequence[-1, 0], n_samples)
    x = np.interp(r, sequence[:, 0], sequence[:, 1])
    return np.stack([r, x], axis=1)


def bernstein_polynomial(t, n):
    poly = []
    for i in range(n + 1):
        poly.append(comb(n, i) * (t ** i) * (1 - t) ** (n - i))
    return np.stack(poly, axis=1)


def bezier_fit(sequence, order=9):
    bounds = [sequence[0], sequence[-1]]
    if (bounds[1] - bounds[0] == 0).any():
        return np.asarray(bounds).ravel(), np.array([0.] * order + [1.])

    unit_seq = (sequence - bounds[0]) / (bounds[1] - bounds[0])
    bern = bernstein_polynomial(unit_seq[:, 0], order)
    bern_inv = np.linalg.pinv(bern)
    params = np.array([0.] * order + [1.])  # 0 is 0 and -1 is 1
    params[1:-1] = bern_inv[1:-1] @ unit_seq[:, 1]
    return np.asarray(bounds).ravel(), params


def bezier_sim(params, n_samples=100):
    t = np.linspace(0, 1, n_samples)
    bern = bernstein_polynomial(t, params.shape[-1] - 1)
    return bern @ params


def generate_data_single(*eq_info, mesh_size=1000, abs_params='', bezier_order=9):
    r, x = sp.symbols('r, x')
    equation, grid_points = sample_points(eq_info[0], eq_info[1], mesh_size=mesh_size, abs_params=abs_params)
    contour_gen = contourpy.contour_generator(*grid_points, name='threaded')
    contours = contour_gen.lines(0)
    # prepare derivative to determine stability
    stability = sp.lambdify((r, x), sp.diff(equation, x), 'numpy')
    # format and resample contours
    contour_segs = []
    for contour in contours:
        split_contours = split_lines(contour)
        for sc in split_contours:
            resample_line(sc, 100)
            fitted_curve = bezier_fit(sc, bezier_order)
            fitted_curve = [stability(*sc[len(sc) // 2]), *fitted_curve[0], *fitted_curve[1]]
            contour_segs.append(fitted_curve)
            # contour_segs.append(resample_line(sc, n_samples))
    return equation, contour_segs


def generate_data(equations, mesh_size=1000, abs_params='', bezier_order=9):
    map_func = partial(generate_data_single, mesh_size=mesh_size, abs_params=abs_params, bezier_order=bezier_order)
    with WorkerPool(n_jobs=12) as pool:
        pool_data = pool.map(map_func, equations,
                             progress_bar=True, progress_bar_options={'desc': 'Generating bifurcations'})
    equations, raw_data = zip(*pool_data)
    equations, raw_data = list(equations), list(raw_data)

    # for i, eq in enumerate(tqdm(equations, desc='Generating data')):
    #     # if (i + 1) % 10 == 0: print('g', i + 1)
    #     # generate contours of bifurcation diagram
    #     equations[i], grid_points = sample_points(eq[0], eq[1], mesh_size=mesh_size, abs_params=abs_params)
    #     contour_gen = contourpy.contour_generator(*grid_points, name='threaded')
    #     contours = contour_gen.lines(0)
    #     # prepare derivative to determine stability
    #     stability = sp.lambdify((r, x), sp.diff(sp.sympify(equations[i]), x), 'numpy')
    #     # format and resample contours
    #     contour_segs = []
    #     for contour in contours:
    #         split_contours = split_lines(contour)
    #         for sc in split_contours:
    #             fitted_curve = bezier_fit(sc, bezier_order)
    #             fitted_curve = [stability(*sc[len(sc) // 2]), *fitted_curve[0], *fitted_curve[1]]
    #             contour_segs.append(fitted_curve)
    #             # contour_segs.append(resample_line(sc, n_samples))
    #     raw_data.append(contour_segs)

    index_data = np.zeros((len(raw_data), 2), dtype=int)
    curve_idx = 0
    for i, eq in enumerate(equations):
        index_data[i] = curve_idx, curve_idx + len(raw_data[i])
        curve_idx += len(raw_data[i])
    curve_data = np.concatenate(raw_data, dtype=np.float32)

    return equations, index_data, curve_data


if __name__ == '__main__':
    # equations = generate_equations(['1', 'x**1', 'x**2', 'x**3'])
    bases = ['a01', 'a02*r',
             'a03*(x+b/4)', 'a04*r*(x+b/4)', 'a05*(x+b/4)**2', 'a06*r*(x+b/4)**2', 'a06*(x+b/4)**3', 'a07*r*(x+b/4)**3',
             'a08*sin(2*c01*(x+b/4))', 'a09*r*sin(2*c02*(x+b/4))',
             'a10*sin(2*c03*r*(x+b/4))', 'a11*r*sin(2*c04*r*(x+b/4))']
    params = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11',
              'b', 'c01', 'c02', 'c03', 'c04']
    equations = generate_equations(bases, params, append='-x**5')
    # print(len(equations))
    # equations = [('a*r*x - b*x**2', 'a b')] * 1
    #
    data = generate_data(equations[:2])
    print(len(equations))
    # points_dataframe = create_points_dataframe(data)
    # shape_dataframe = create_shape_dataframe(data, equations)
    # dataframe = points_dataframe.merge(shape_dataframe, on='shape')
    # dataframe.to_feather('data/transcritical_data.feather')
