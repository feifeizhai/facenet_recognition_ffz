import tensorflow as tf
import numpy as np
import cv2
from utils import facenet
import os
from os.path import join
import sys
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from utils.db_manager import Person
import json

def classifier_model(embaddings,labels,names):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = SVC(kernel='poly', degree=4, gamma=1, coef0=0, probability=True)
            model.fit(embaddings, labels)
            classifier_filename_exp = os.path.expanduser('./models/face_classifier_model.pkl')
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, names), outfile)

            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            predict = model.predict(embaddings)
            accuracy = metrics.accuracy_score(labels, predict)
            print('accuracy: %.2f%%' % (100 * accuracy))




def facedata_from_db():

    embaddings = [np.zeros(512)]
    names = ['unknow']
    labels = ['unknow']
    persons = Person.select()
    for person in persons:
        names.append(person.user_name)
        face_datas = json.loads(person.face_data)
        if face_datas is None:
            continue

        for face_data in face_datas:
            face_embs = face_data['face_data']
            for data in face_embs:
                embaddings.append(data)
                labels.append(person.user_name)

    return embaddings, names, labels


def classify():

    embaddings, names, labels = facedata_from_db()
    classifier_model(embaddings, labels, names)

if __name__ == '__main__':
    embaddings, names, labels = facedata_from_db()
    classifier_model(embaddings, labels, names)