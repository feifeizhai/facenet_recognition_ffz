
from utils import facenet
import pickle

import tensorflow as tf

from face_align import mtcnn
import os
import re
import sys

import numpy as np
from utils import tools


def getcwd():

    path = sys.argv[0]
    # path = str(path).index('facenet_server.py','')
    path = os.path.split(path)[0]
    # print(path)
    return path

class facenetEmbedding:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        base_path = getcwd()
        dirName1 = "models"
        dirName2 = "facenet_model"
        model_name = "20180402-114759.pb"
        model_path = os.path.normpath("%s/%s/%s/%s"%( base_path, dirName1, dirName2,model_name))
        print(model_path)
        # model_path=r'./models/facenet_model/20180402-114759.pb'

        # Load the models
        facenet.load_model(model_path)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.tf_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def  get_embedding(self,images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.tf_embeddings, feed_dict=feed_dict)

        return embedding
    def free(self):
        self.sess.close()


class MTCNN():
    def __init__(self):
        self.minsize = 155 # minimum size of face
        self.threshold = [0.6, 0.7, 0.8]  # three steps's threshold
        self.factor = 0.709  # scale factor
        base_path = getcwd()
        dirName1 = "models"
        dirName2 = "mtcnn_model"

        model_path = os.path.normpath("%s/%s/%s" % (base_path, dirName1, dirName2))
        print(model_path)
        file_paths = self.get_model_filenames(model_path)
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():

            sess = tf.Session()
            with sess.as_default():

                self.pnet, self.rnet, self.onet = mtcnn.create_mtcnn(sess, file_paths)

    def get_model_filenames(self, model_dir):
        # print(os.getcwd())
        # print(os.listdir(os.getcwd()))
        files = os.listdir(model_dir)
        pnet = [s for s in files if 'pnet' in s and
                os.path.isdir(os.path.join(model_dir, s))]
        rnet = [s for s in files if 'rnet' in s and
                os.path.isdir(os.path.join(model_dir, s))]
        onet = [s for s in files if 'onet' in s and
                os.path.isdir(os.path.join(model_dir, s))]
        if pnet and rnet and onet:
            if len(pnet) == 1 and len(rnet) == 1 and len(onet) == 1:
                _, pnet_data = self.get_meta_data(os.path.join(model_dir, pnet[0]))
                _, rnet_data = self.get_meta_data(os.path.join(model_dir, rnet[0]))
                _, onet_data = self.get_meta_data(os.path.join(model_dir, onet[0]))
                return (pnet_data, rnet_data, onet_data)
            else:
                raise ValueError('There should not be more '
                                 'than one dir for each models')
        else:
            return self.get_meta_data(model_dir)

    def get_meta_data(self, model_dir):

        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the models '
                             'directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than '
                             'one meta file in the models directory (%s)'
                             % model_dir)
        meta_file = meta_files[0]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^[A-Za-z]+-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    data_file = step_str.groups()[0]
        return (os.path.join(model_dir, meta_file),
                os.path.join(model_dir, data_file))


    def detect_face(self,image,fixed=None):
        '''
        mtcnn人脸检测，
        PS：人脸检测获得bboxes并不一定是正方形的矩形框，参数fixed指定等宽或者等高的bboxes
        :param image:
        :param fixed:
        :return:
        '''
        bboxes, landmarks = tools.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        landmarks_list = []
        landmarks=np.transpose(landmarks)
        bboxes=bboxes.astype(int)
        bboxes = [b[:4] for b in bboxes]
        for landmark in landmarks:
            # face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(landmark)
        if fixed is not None:
            bboxes,landmarks_list=self.get_square_bboxes(bboxes, landmarks_list, fixed)
        return bboxes,landmarks_list

    def get_square_bboxes(self, bboxes, landmarks, fixed="height"):
        '''
        获得等宽或者等高的bboxes
        :param bboxes:
        :param landmarks:
        :param fixed: width or height
        :return:
        '''
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w = (x2 - x1)
            h = (y2 - y1)

            if fixed == "height":
                dd = h / 2
            elif fixed == 'width':
                dd = w / 2
            center_x, center_y = (int((x1 + x2) / 2), int((y1 + y2) / 2 + 0.2 * dd))
            x11 = int(center_x - dd)
            y11 = int(center_y - dd)
            x22 = int(center_x + dd)
            y22 = int(center_y + dd)
            new_bbox = (x11, y11, x22, y22)
            new_bboxes.append(new_bbox)
        return new_bboxes, landmarks


class Classifier:
    def __init__(self):
        base_path = getcwd()
        dirName1 = "models"

        model_name = "face_classifier_model.pkl"
        model_path = os.path.normpath("%s/%s/%s" % (base_path, dirName1, model_name))
        classifier_filename_exp = os.path.expanduser(model_path)
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_names) = pickle.load(infile)


    def predict(self, emb_array):

        predictions = self.model.predict_proba(emb_array)

        predict = self.model.predict(emb_array)

        return predict, predictions

