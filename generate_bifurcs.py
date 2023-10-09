# import PyDSTool as dst
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations
from string import ascii_lowercase

from matplotlib.contour import QuadContourSet
from scipy import ndimage
import contourpy
from scipy.signal import find_peaks


def generate_equations(bases, append=None):
    par_bases = bases + [f'r * {s}' for s in bases]
    var_sets = [s for i in range(1, len(par_bases) + 1) for s in combinations(par_bases, i)]
    for b in bases:
        var_sets = [v for v in var_sets if sum([b in sv for sv in v]) < 2]  # remove duplicates
    var_sets = [v for v in var_sets if sum(['r' in sv for sv in v]) > 0]  # ensure r is present
    equations = []
    for v in var_sets:
        defining_pars = ascii_lowercase[:len(v)]
        equation = ' + '.join([f'{p}*{s}' for p, s in zip(defining_pars, v)])
        if append is not None:
            equation = equation + ' + ' + append
        # equation = equation + ' - 0.01*x**5'  # ensure bounded solutions
        equations.append((equation, defining_pars))
    return equations


# # currently only supports 1D systems
# def setup_system(equation, params, param_values=None):
#     # add r to params
#     params = 'r' + params
#     if param_values is None:
#         param_values = np.random.randn(len(params)).clip(-2, 2)
#
#     ds_args = dst.args(name='Model')
#     ds_args.pars = {p: v for p, v in zip(params, param_values)}
#     ds_args.varspecs = {'x': equation, 'w': 'x - w'}
#     ds_args.ics = {'x': 0, 'w': 0}
#     ds_args.tdomain = [0, 100]  # set the range of integration.
#     ds_args.pdomain = {p: [-2, 2] for p in params}
#     ds_args.xdomain = {'x': [-10, 10], 'w': [-np.inf, np.inf]}
#     return dst.Generator.Vode_ODEsystem(ds_args)
#
#
# def generate_bifurcation(ode):
#     # set initial conditions near fixed point
#     traj = ode.compute('trajectory')  # integrate ODE
#     fixed = traj.sample(dt=0.1)['x'][-1]
#     ode.set(ics={'x': fixed + 0.01})
#
#     # set up initial continuation class
#     pc_args = dst.args(name='EQ1', type='EP-C')
#     pc_args.freepars = ['r']
#     pc_args.StepSize = 1e-3
#     pc_args.MaxStepSize = 1e-2
#     pc_args.MinStepSize = 1e-4
#     pc_args.MaxNumPoints = 10000
#     pc_args.LocBifPoints = 'all'
#     pc_args.StopAtPoints = ['B']  # stop at boundary points
#     pc_args.SaveEigen = True  # to tell unstable from stable branches
#
#     # compute bifurcation diagram
#     pc = dst.ContClass(ode)
#     pc.newCurve(pc_args)
#     pc['EQ1'].forward()
#     pc['EQ1'].backward()
#
#
#     pc.display(['r', 'x'], stability=True)
#     plt.xlim(-2, 2)
#     plt.ylim(-10, 10)
#     plt.show()

def sample_points(equation, params, param_values=None, abs_params='', mesh_size=100):
    if param_values is None:
        param_values = np.random.randn(len(params)).clip(-1, 1)
    for p, pv in zip(params, param_values):
        if p in abs_params: pv = abs(pv)
        equation = equation.replace(p, str(pv))

    # print(equation)

    # @np.vectorize
    def f(r, x):
        return eval(equation)

    points = np.meshgrid(np.linspace(-1, 1, mesh_size), np.linspace(-5, 5, mesh_size))
    # res = QuadContourSet(points[0], points[1], f(points[0], points[1]), levels=[0])
    # plt.xlabel('r')
    # plt.ylabel('x')
    # plt.show()
    return equation, (points[0], points[1], f(points[0], points[1]))


def split_lines(sequence, deadzone=1):
    # split sequence into monotonic segments
    smoothed = ndimage.gaussian_filter1d(sequence.copy().T, 3)
    second_diff = np.abs(np.diff(np.diff(smoothed)))
    try:
        peaks_r = find_peaks(second_diff[0], prominence=second_diff[0].max()/2)[0]
    except ValueError:
        peaks_r = []
    try:
        peaks_x = find_peaks(second_diff[1], prominence=second_diff[1].max()/2)[0]
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


def create_points_dataframe(segments):
    points_data = []
    for i, shape in enumerate(segments):
        for j, segment in enumerate(shape):
            for k, point in enumerate(segment):
                points_data.append({
                    'shape': i,
                    'segment': j,
                    'point': k,
                    'r': point[0],
                    'x': point[1]
                })
    dataframe = pd.DataFrame(points_data)
    return dataframe


def create_shape_dataframe(segments, equations):
    shape_data = []
    for i, (shape, eq) in enumerate(zip(segments, equations)):
        shape_data.append({
            'shape': i,
            'n_segments': len(shape),
            'equation': eq,
        })
    dataframe = pd.DataFrame(shape_data)
    return dataframe


def generate_data(equations, n_samples=100, abs_params=''):
    data = []
    for i, eq in enumerate(equations):
        if (i + 1) % 10 == 0: print('g', i + 1)
        # generate contours of bifurcation diagram
        equations[i], grid_points = sample_points(eq[0], eq[1], mesh_size=1000, abs_params=abs_params)
        contour_gen = contourpy.contour_generator(*grid_points, name='threaded')
        contours = contour_gen.lines(0)
        # format and resample contours
        contour_segs = []
        for contour in contours:
            split_contours = split_lines(contour)
            for sc in split_contours:
                contour_segs.append(resample_line(sc, n_samples))
        data.append(contour_segs)
    return data

def generate_taylor_data(equations, terms=10, abs_params=''):
    data = []
    for i, eq in enumerate(equations):
        # if (i + 1) % 10 == 0: print(i + 1)
        # generate contours of bifurcation diagram
        equations[i], grid_points = sample_points(eq[0], eq[1], mesh_size=1000, abs_params=abs_params)
        contour_gen = contourpy.contour_generator(*grid_points, name='threaded')
        contours = contour_gen.lines(0)
        # format and resample contours
        contour_segs = []
        for contour in contours:
            split_contours = split_lines(contour)
            for sc in split_contours:
                contour_segs.append(resample_line(sc, n_samples))
        data.append(contour_segs)
    return data


if __name__ == '__main__':
    # equations = generate_equations(['1', 'x**1', 'x**2', 'x**3'])
    equations = [('a*r*x - b*x**2', 'ab')] * 1000
    data = generate_data(equations)
    print(data)
    points_dataframe = create_points_dataframe(data)
    shape_dataframe = create_shape_dataframe(data, equations)
    dataframe = points_dataframe.merge(shape_dataframe, on='shape')
    dataframe.to_feather('data/transcritical_data.feather')
