import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from generate_bifurcs import unpack_bifurc_vector


def plot_bifurcation(vectors, color='k', show=True, lim=((-1, 1), (-2, 2)), fmt='mathematica'):
    unpacked = []
    if fmt == 'python':
        for vec in vectors:
            u, r, x = unpack_bifurc_vector(vec)
            unpacked.append((u, r, x))
            if u > 0.5:
                plt.plot(r, x, color=color, linestyle='--')
            else:
                plt.plot(r, x, color=color)
        plt.xlim(*lim[0])
        plt.ylim(*lim[1])
    elif fmt == 'mathematica':
        if not hasattr(color, '__iter__') or isinstance(color, str):
            color = [color] * len(vectors[0])
        for i, (box, stab, branch) in enumerate(zip(*vectors)):
            r = np.linspace(*box[:2], len(branch))
            branch = branch * (box[3] - box[2]) + box[2]
            if stab < 0.5:
                plt.plot(r, branch, color=color[i], linestyle='--')
            else:
                plt.plot(r, branch, color=color[i])
        plt.xlim(*lim[0])
        plt.ylim(*lim[1])
    if show:
        plt.show()
    return unpacked


def decode_results(box, branch):
    r = np.linspace(*box[:2], len(branch))
    branch = np.asarray(branch * (box[3] - box[2]) + box[2])
    return np.stack([r, branch]).T


def bifurc_nms(results, distance_metric=None, threshold=0.2, dist_scale=0.5):
    scores, box, stab, branch = results
    scores = torch.sigmoid(scores.squeeze())
    # stab = torch.sigmoid(stab)
    traj = np.stack([decode_results(bo, br) for bo, br in zip(box, branch)])

    mask = scores > threshold
    traj = traj[mask]
    if len(traj) == 0:
        return box[mask], stab[mask], branch[mask]

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

    print(np.arange(len(box))[mask][valid])
    return box[mask][valid], stab[mask][valid], branch[mask][valid]


# def get_bifurcation_assignment(prediction, target):

def bifurcation_proximity(prediction, target, distance_metric=None, dist_scale=0.5):
    pred_traj = [decode_results(box, branch) for box, stab, branch in zip(*prediction)]
    targ_traj = [decode_results(box, branch) for box, stab, branch in zip(*target)]

    if len(targ_traj) == 0 or len(pred_traj) == 0:
        return np.zeros(1)

    cost_matrix = np.zeros((len(pred_traj), len(targ_traj)))
    for i in range(len(pred_traj)):
        for j in range(len(targ_traj)):
            dist = distance_metric(pred_traj[i], targ_traj[j]) * dist_scale
            cost_matrix[i, j] = dist.clip(max=1)
    assignment = linear_sum_assignment(cost_matrix)

    distance = 0
    for i, j in zip(*assignment):
        distance += np.sqrt(np.sum((pred_traj[i][[0, -1]] - targ_traj[j][[0, -1]]) ** 2, axis=-1)).mean()
    return distance / min(len(pred_traj), len(targ_traj))

def bifurcation_similiarity(prediction, target, distance_metric=None, dist_scale=0.5):
    pred_traj = [decode_results(box, branch) for box, stab, branch in zip(*prediction)]
    targ_traj = [decode_results(box, branch) for box, stab, branch in zip(*target)]

    if len(targ_traj) == 0:
        if len(pred_traj) == 0:
            return np.zeros(1)
        else:
            return np.ones(1)

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

# def bifurcation_similiarity(prediction, target, distance_metric=None, dist_scale=0.5):
#     pred_traj = [np.stack(unpack_bifurc_vector(r)[1:]) for r in prediction]
#     targ_traj = [np.stack(unpack_bifurc_vector(r)[1:]) for r in target]
#     cost_matrix = np.zeros((len(pred_traj), len(targ_traj)))
#     for i in range(len(pred_traj)):
#         for j in range(len(targ_traj)):
#             dist = distance_metric(pred_traj[i], targ_traj[j]) * dist_scale
#             cost_matrix[i, j] = dist.clip(max=1)
#
#     assignment = linear_sum_assignment(cost_matrix)
#     distance = 0
#     for i, j in zip(*assignment):
#         distance += cost_matrix[i, j]
#     for i in range(len(pred_traj)):
#         if i not in assignment[0]:
#             distance += cost_matrix[i].min()
#     distance = (distance + len(targ_traj) - len(assignment[1])) / len(targ_traj)
#
#     return np.clip(distance, 0, 1)
