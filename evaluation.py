import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from generate_bifurcs import unpack_bifurc_vector


def plot_bifurcation(vectors, color='k', show=True, lim=((-1, 1), (-2, 2))):
    unpacked = []
    for vec in vectors:
        u, r, x = unpack_bifurc_vector(vec)
        unpacked.append((u, r, x))
        if u > 0.5:
            plt.plot(r, x, color=color, linestyle='--')
        else:
            plt.plot(r, x, color=color)
    plt.xlim(*lim[0])
    plt.ylim(*lim[1])
    if show:
        plt.show()
    return unpacked


def bifurc_nms(results, distance_metric=None, threshold=0.2, dist_scale=0.5):
    scores = torch.sigmoid(results[:, 0])
    mask = scores > threshold
    traj = [np.stack(unpack_bifurc_vector(r)[1:]) for r in results[mask]]
    if len(traj) == 0:
        return results[mask, 1:][[]]

    distance = np.zeros((len(traj), len(traj)))
    traj = np.stack(traj)
    for i in range(len(traj)):
        for j in range(i + 1, len(traj)):
            # distance[i, j], _ = sm.dtw(traj[i], traj[j])
            distance[i, j] = distance_metric(traj[i], traj[j])
            distance[j, i] = distance[i, j]
    distance = (distance * dist_scale).clip(0, 1)

    valid = []
    scores = scores[mask].clone()
    s, si = torch.max(scores, 0)
    while s > threshold:
        valid.append(si.item())
        scores = scores * distance[si]
        s, si = torch.max(scores, 0)

    return results[mask, 1:][valid]

# def get_bifurcation_assignment(prediction, target):

def bifurcation_similiarity(prediction, target, distance_metric=None, dist_scale=0.5):
    pred_traj = [np.stack(unpack_bifurc_vector(r)[1:]) for r in prediction]
    targ_traj = [np.stack(unpack_bifurc_vector(r)[1:]) for r in target]
    cost_matrix = np.zeros((len(pred_traj), len(targ_traj)))
    for i in range(len(pred_traj)):
        for j in range(len(targ_traj)):
            dist = distance_metric(pred_traj[i], targ_traj[j]) * dist_scale
            cost_matrix[i, j] = dist.clip(max=1)

    assignment = linear_sum_assignment(cost_matrix)
    distance = 0
    for i, j in zip(*assignment):
        distance += cost_matrix[i, j]
    for i in range(len(pred_traj)):
        if i not in assignment[0]:
            distance += cost_matrix[i].min()
    distance = (distance + len(targ_traj) - len(assignment[1])) / len(targ_traj)

    return np.clip(distance, 0, 1)
