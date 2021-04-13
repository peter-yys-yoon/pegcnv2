import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import angle_between, rotation_matrix

class NTU(Dataset):
    def __init__(self, type='train', setting='cs', data_shape=(3,300,25,2), transform=None):

        self.maxC, self.maxT, self.maxV, self.maxM = data_shape
        self.transform = transform

        file = './datasets/' + setting + '_' + type + '.txt'
        if not os.path.exists(file):
            raise ValueError('Please generate data first! Using gen_data.py in the main folder.')

        fr = open(file, 'r')
        self.files = fr.readlines()
        fr.close()

    def pre_normalization(self, data, zaxis=[0, 1], xaxis=[8, 4]):
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

        # print('pad the null frames with the previous frames')
        for i_s, skeleton in enumerate(s):  # pad
            if skeleton.sum() == 0:
                print(i_s, ' has no skeleton')
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                if person[0].sum() == 0:
                    index = (person.sum(-1).sum(-1) != 0)
                    tmp = person[index].copy()
                    person *= 0
                    person[:len(tmp)] = tmp
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        if person[i_f:].sum() == 0:
                            rest = len(person) - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                            s[i_s, i_p, i_f:] = pad
                            break

        # print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            main_body_center = skeleton[0][:, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

        # print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            joint_bottom = skeleton[0, 0, zaxis[0]]
            joint_top = skeleton[0, 0, zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

        # print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            joint_rshoulder = skeleton[0, 0, xaxis[0]]
            joint_lshoulder = skeleton[0, 0, xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

        data = np.transpose(s, [0, 4, 2, 3, 1])
        return data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx].strip()
        label = file_name.split('/')[-1]
        label = int(label.split('A')[1][:3]) - 1

        data = np.zeros((self.maxC, 300, self.maxV, self.maxM))
        location = np.zeros((2, self.maxT, self.maxV, self.maxM))
        with open(file_name, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                if frame >= self.maxT:
                    break
                person_num = int(fr.readline())
                for person in range(person_num):
                    fr.readline()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        v = fr.readline().split(' ')
                        if joint < self.maxV and person < self.maxM:
                            data[0,frame,joint,person] = float(v[0])
                            data[1,frame,joint,person] = float(v[1])
                            data[2,frame,joint,person] = float(v[2])
                            location[0,frame,joint,person] = float(v[5])
                            location[1,frame,joint,person] = float(v[6])

        if frame_num <= self.maxT:
            data = data[:,:self.maxT,:,:]
        else:
            s = frame_num // self.maxT
            r = random.randint(0, frame_num - self.maxT * s)
            new_data = np.zeros((self.maxC, self.maxT, self.maxV, self.maxM))
            for i in range(self.maxT):
                new_data[:,i,:,:] = data[:,r+s*i,:,:]
            data = new_data

        data =np.expand_dims(data, 0)
        data = self.pre_normalization(data)[0]
        if self.transform:
            (data, location) = self.transform((data, location))

        data = torch.from_numpy(data).float()
        location = torch.from_numpy(location).float()
        label = torch.from_numpy(np.array(label)).long()
        return data, location, label, file_name
