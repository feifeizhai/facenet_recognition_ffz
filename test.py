import cv2
from face_collect import face_collect
from face_identify import face_identify
import json
import time
model_path='models/20180402-114759.pb'
dataset_path='dataset/emb/faceEmbedding.npy'
filename='dataset/emb/name.txt'
image_path='dataset/test_images/1.jpg'

cap = cv2.VideoCapture(0)


def image2data_image(collect,image=None, real_name = None,user_name = None):

    # if collect.gesture_count == 0:
    #     print("请注视摄像头")
    # if collect.gesture_count == 1:
    #     print("请缓慢低头")
    # if collect.gesture_count == 2:
    #     print("向右缓慢转头")
    # if collect.gesture_count == 2:
    #     print("向左缓慢转头")

    pred_emb = collect.create_face_vector(image, real_name = real_name, user_name = user_name)

        # predict.saver_data_to_csv(pred_emb, name)
    return pred_emb



def image2data_video(collect, real_name = None,user_name = None):
    while cap.isOpened():
        ret, image = cap.read()
        # image = cv2.imread('./bill.jpg')
        result = image2data_image(collect, image, real_name, user_name)
        # print(result)
        if result == True:
            collect_frame_to_csv()
            # cap.release()
            # cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

def collect_frame_to_csv():
    choos=input("是否录入人脸信息(y/n)")
    if choos=='y':
        real_name = input("请输入真实姓名:")
        user_name = input("请输入用户名:")
        collect = face_collect()
        image2data_video(collect,real_name,user_name)
        # collect_frame_to_csv();

    else:
        dentify = face_identify()
        compare_video(dentify)
        print('go next steps...')

def compare_video(dentify):
    while cap.isOpened():
        ret, image = cap.read()
        compare_image(dentify,image)

    cap.release()
    cv2.destroyAllWindows()

def compare_image(dentify,image):
    try:

        result = dentify.face_recognition_image(image)
        count = '{}'.format(len(result))
        dic = {'AppType': '1','LocalPath': json.dumps({'count': count, 'names': result})}
        resultstr = json.dumps(dic)

        return resultstr

    except:

        dic = {'AppType': '1', 'error':'图片获取出错'}
        resultstr = json.dumps(dic)
        return resultstr



if __name__ == '__main__':

    collect_frame_to_csv()