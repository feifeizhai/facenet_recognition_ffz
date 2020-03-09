from socket import *

from face_identify import face_identify
import cv2
import os
import json

from image_similar import image_similar
from utils.db_manager import Person
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import sklearn.utils._cython_blas
import tensorflow as tf


HOST = ''
PORT = 21567
BUFSIZ = 1024
ADDR = (HOST,PORT)

tcpSerSock = socket(AF_INET,SOCK_STREAM)
tcpSerSock.bind(ADDR)
tcpSerSock.listen(5)

dentify = face_identify()

def image_simil(image_path):

    image_path = image_path.split(',')

    result = image_similar.compare_images(image_path[0], image_path[1])

    return result
def face_server(image_path):


    img = cv2.imread(image_path)
    result = compare_face(img)
    # 判断文件是否存在
    if (os.path.exists(image_path)):
        os.remove(image_path)
        print('移除后test 目录下有文件：%s' % image_path)
    else:
        print('要删除的文件不存在！')
    return result

def compare_face(image):
    try:
        pred_real_name,pred_user_name = dentify.face_recognition_image(image)
        count = '{}'.format(len(pred_user_name))
        dic = {'AppType': '1','LocalPath': json.dumps({'count': count, 'real_name': pred_real_name,'user_name': pred_user_name})}
        resultstr = json.dumps(dic)
        return resultstr

    except:

        dic = {'AppType': '1', 'error':'图片获取出错'}
        resultstr = json.dumps(dic)
        return resultstr

while True:
    print('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('...connnecting from:', addr)

    while True:
        try:
            data = tcpCliSock.recv(BUFSIZ)
            json_response = data.decode('utf-8')
            print('接收 = ' + json_response)
        except:
            json_response = data.decode('utf-8')
            print('接收错误 = ' + json_response)
            break
        # data.decode('utf-8')
        # img = cv2.imread(r'C:\Users\jingge\Desktop\Face_Detection_Recognition-master\faceRecognition\dataset\images\192.168.1.100_01_20190820135414548.jpg')

        try:
            dict_json = json.loads(json_response)
            type = dict_json['AppType']
            path = dict_json['LocalPath']
        except:
            print('解析错误' + json_response)
            break

        if int(type) == 1:
            result = face_server(path)
        elif int(type) == 2:
            result = image_simil(path)
        else:
            result = '没找到{}类型'.format(type)
            print(result)
            break

        if not data:
            break
        try:
            print('发送 = ' + result)
            tcpCliSock.send(result.encode())
        except:
            print('发送错误 = ' + result)
            break
        #tcpCliSock.send('[%s] %s' %(bytes(ctime(),'utf-8'),data))

    tcpCliSock.close()
tcpSerSock.close()