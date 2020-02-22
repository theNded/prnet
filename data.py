#!/usr/bin/env python
# -*- coding: utf-8 -*-
import open3d as o3d
import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNet40, self).__init__()
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


from scipy.linalg import expm, norm
from util import npmat2euler
import MinkowskiEngine as ME
from util import npmat2euler

def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(
        randg.rand(3) - 0.5,
        rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class IndoorPairDataset(Dataset):
    OVERLAP_RATIO = None
    AUGMENT = None

    def __init__(self,
                 dataset_path,
                 phase,
                 num_points=4096,
                 random_rotation=True,
                 rotation_range=360,
                 voxel_size=0.025,
                 manual_seed=False):
        self.files = []

        self.num_points = num_points
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.voxel_size = voxel_size
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        self.root = root = dataset_path
        self.phase = phase
        print("Loading the subset {} from {}".format(phase, root))

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(fnames_txt
                       ) > 0, "Make sure that the path {} has data {}".format(
                           root, fname)

            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def reset_seed(self, seed=0):
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelization and sampling
        sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size,
                                        return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size,
                                        return_index=True)
        xyz0 = xyz0[np.random.choice(sel0, self.num_points)]
        xyz1 = xyz1[np.random.choice(sel1, self.num_points)]

        pointcloud1 = xyz0.T
        pointcloud2 = xyz1.T

        inv_trans = np.linalg.inv(trans)
        R_ab = trans[:3, :3]
        translation_ab = trans[:3, 3]
        euler_ab = npmat2euler(np.expand_dims(R_ab, 0))

        R_ba = inv_trans[:3, :3]
        translation_ba = inv_trans[:3, 3]
        euler_ba = npmat2euler(np.expand_dims(R_ba, 0))

        # Selection
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')


class ThreeDMatchPairDataset03(IndoorPairDataset):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        'train': '../../config/train_3dmatch.txt',
        'val': '../../config/val_3dmatch.txt',
        'test': '../../config/test_3dmatch.txt'
    }


class ThreeDMatchPairDataset05(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.5


class ThreeDMatchPairDataset07(ThreeDMatchPairDataset03):
    OVERLAP_RATIO = 0.7


if __name__ == '__main__':
    train = ThreeDMatchPairDataset03(
        '/home/wei/Workspace/data/threedmatch',
        phase='val',
        num_points=1024,
        random_rotation=True,
        rotation_range=360,
        voxel_size=0.05,
        manual_seed=False)

    # train = ModelNet40(1024)

    for data in train:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(data[0].T.astype('float64'))

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(data[1].T.astype('float64'))

        o3d.visualization.draw_geometries([pcd0, pcd1])

        # a->b
        Tab = np.eye(4)
        Tab[:3, :3] = data[2]
        Tab[:3, 3] = data[3]

        pcd0.transform(Tab)
        o3d.visualization.draw_geometries([pcd0, pcd1])
        pcd0.transform(np.linalg.inv(Tab))

        # b->a
        Tba = np.eye(4)
        Tba[:3, :3] = data[4]
        Tba[:3, 3] = data[5]

        pcd1.transform(Tba)
        o3d.visualization.draw_geometries([pcd0, pcd1])
        pcd1.transform(np.linalg.inv(Tba))
