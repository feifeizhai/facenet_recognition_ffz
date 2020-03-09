import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from math import cos, sin
from . predictor import PosPrediction

class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, './models/PRNet_model/256_256_resfcn256_weight')
        # if not os.path.isfile(prn_path + '.data-00000-of-00001'):
        #     print("please download PRN trained models first.")
        #     exit()
        self.pos_predictor.restore(prn_path)

        # uv file

        self.uv_kpt_ind = np.loadtxt('./models/PRNet_model/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt('./models/PRNet_model/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt('./models/PRNet_model/triangles.txt').astype(np.int32) # ntri x 3

        self.uv_coords = self.generate_uv_coords()

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords



    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors

    # import scipy.io as
    def frontalize(self, vertices):
        canonical_vertices = np.load('./PRNet/Data/uv-data/canonical_vertices.npy')

        vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0], 1])))  # n x 4
        P = np.linalg.lstsq(vertices_homo, canonical_vertices, rcond=-1)[0].T  # Affine matrix. 3 x 4
        front_vertices = vertices_homo.dot(P.T)

        return front_vertices

    def rotate(self, vertices, angles):
        ''' rotate vertices.
        X_new = R.dot(X). X: 3 x 1
        Args:
            vertices: [nver, 3].
            rx, ry, rz: degree angles
            rx: pitch. positive for looking down
            ry: yaw. positive for looking left
            rz: roll. positive for tilting head right
        Returns:
            rotated vertices: [nver, 3]
        '''

        R = self.angle2matrix(angles)
        rotated_vertices = vertices.dot(R.T)

        return rotated_vertices

    def angle2matrix(self, angles):
        ''' get rotation matrix from three rotation angles(degree). right-handed.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch. positive for looking down.
            y: yaw. positive for looking left.
            z: roll. positive for tilting head right.
        Returns:
            R: [3, 3]. rotation matrix.
        '''
        x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
        # x
        Rx = np.array([[1, 0, 0],
                       [0, cos(x), -sin(x)],
                       [0, sin(x), cos(x)]])
        # y
        Ry = np.array([[cos(y), 0, sin(y)],
                       [0, 1, 0],
                       [-sin(y), 0, cos(y)]])
        # z
        Rz = np.array([[cos(z), -sin(z), 0],
                       [sin(z), cos(z), 0],
                       [0, 0, 1]])

        R = Rz.dot(Ry.dot(Rx))
        return R.astype(np.float32)