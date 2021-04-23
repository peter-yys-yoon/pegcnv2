import os
import shutil
import pynvml
import torch
import numpy as np
import math
import time
import yaml


def check_gpu(gpus):
    if len(gpus) > 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        for i in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memused = meminfo.used / 1024 / 1024
            print('GPU{} used: {}M'.format(i, memused))
            if memused > 5000:
                pynvml.nvmlShutdown()
                raise ValueError('Mem used {}, GPU{} is occupied!'.format(memused, i))
        pynvml.nvmlShutdown()
        return torch.device('cuda')
    else:
        print('Using CPU!')
        return torch.device('cpu')


def load_checkpoint(tag, fname='checkpoint'):
    fpath = f'./models/{tag}/' + fname + '.pth.tar'
    print('>>>>>> loading', fpath)
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath)
        # import pdb
        # pdb.set_trace()
        return checkpoint
    else:
        raise ValueError('Do NOT exist this checkpoint: {}'.format(fname))


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    export_path = f'./models/{args.tag}'

    if not os.path.exists(export_path):
        os.makedirs(export_path)
    with open('{}/config.yaml'.format(export_path), 'w') as f:
        yaml.dump(arg_dict, f)


def print_log_train(tag, given, print_time=True):
    export_path = f'./models/{tag}/train_log.txt'
    timestamp = time.strftime("[%m.%d.%y|%X] ", time.localtime())
    with open(export_path, 'a') as f:
        print(timestamp + ' ' + given, file=f)


def print_log_eval(tag, given, print_time=True):
    export_path = f'./models/{tag}/eval_log.txt'

    timestamp = time.strftime("[%m.%d.%y|%X] ", time.localtime())

    aa = [timestamp] + given
    with open(export_path, 'a') as f:
        print(aa.joint(';'), file=f)


def save_checkpoint(model, optimizer, epoch, best, is_best, model_name, tag):
    export_path = f'./models/{tag}'
    if not os.path.exists(export_path):
        os.mkdir(export_path)
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {'model': model, 'optimizer': optimizer, 'epoch': epoch, 'best': best}
    torch.save(checkpoint, export_path + '/checkpoint.pth.tar')
    if is_best:
        shutil.copy(export_path + '/checkpoint.pth.tar', export_path + '/' + model_name + '.pth.tar')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot recognize the input parameter {}'.format(v))


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
