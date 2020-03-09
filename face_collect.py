import cv2
import json
import numpy as np

from utils.db_manager import Person
from utils import image_processing
from utils import tools
from utils import face_recognition
from utils import classifier_model

from PRNet.api import PRN
from PRNet import render
from PRNet import estimate_pose

resize_width = 160
resize_height = 160



class face_collect():

    def __init__(self):
        self.face_detect = face_recognition.MTCNN()
        self.face_net = face_recognition.facenetEmbedding()
        self.prn = PRN()
        self.bg_list = []
        self.gesture_count = 0

    def face_rect(self, image=None):

        bboxes, landmarks = self.face_detect.detect_face(image, fixed="height")

        return bboxes, landmarks

    def face_512_vector(self,face_images=None):

        face_images = image_processing.get_prewhiten_images(face_images)
        pred_emb = self.face_net.get_embedding(face_images)
        return pred_emb

    def CLAHE(self,img, clipLimit=2.0,
              tileGridSize=(4, 4)):
        # img = img.astype(np.float32)

        clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                tileGridSize=tileGridSize)
        new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_bilateral_image = cv2.bilateralFilter(new_image, 4, 75, 75)

        # 限制对比度的自适应阈值均衡化
        new_image = clahe.apply(new_image)
        new_bilateral_image = clahe.apply(new_bilateral_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        new_bilateral_image = cv2.cvtColor(new_bilateral_image, cv2.COLOR_GRAY2BGR)
        return new_image, new_bilateral_image

    def extension_img(self,bg_list=None):

        person = Person()
        euc_dists = []
        cos_dists = []
        image_list = []
        print('数据处理中,请稍等...')
        for image in bg_list:

            [w, h, c] = image.shape
            for scale in np.arange(0.4, 1.7, 0.6):
                bg_image = image_processing.resize_image(image, int(w * scale), int(h * scale))

                for angle in np.arange(-30, 31, 15):
                    rotate_bg_image = tools.rotate_bound(bg_image, angle)

                    bboxes, landmarks = self.face_rect(rotate_bg_image)

                    if len(bboxes) == 0:
                        print("-----no face")
                    else:

                        new_images = image_processing.get_bboxes_image(rotate_bg_image, bboxes, landmarks,
                                                                       resize_width,
                                                                       resize_height)
                        # for clipLimit in np.arange(0.5, 3, 0.5):

                        new_image, bilateral_image = self.CLAHE(new_images[0], clipLimit=2)
                        image_list.append(new_image)
                        image_list.append(bilateral_image)
                        new_clahe_image, clahe_bilateral_image = self.CLAHE(np.fliplr(new_images[0]))
                        image_list.append(new_clahe_image)
                        image_list.append(clahe_bilateral_image)
                        cv2.imshow("789", clahe_bilateral_image)
                        cv2.waitKey(1)

        image_emb = self.face_512_vector(image_list)
        face_data = image_emb.tolist()
        person.face_data = [{'yaw': '{}'.format(0), 'pitch': '{}'.format(0), 'face_data': face_data}]
        person.euc_dists = euc_dists
        person.cos_dists = cos_dists
        print('模型生成中,请稍等...')
        return person

    def live_test(self, image):



        try:
            bboxes, landmarks = self.face_rect(image)
            images = image_processing.get_bboxes_image(image, bboxes, landmarks,
                                                   256,
                                                   256)
            face = images[0]
            prn_face = face / 255.
            pos = self.prn.net_forward(prn_face)

            vertices = self.prn.get_vertices(pos)

            camera_matrix, pose = estimate_pose.estimate_pose(vertices)
            l_r, u_d, _ = pose[0], pose[1], pose[2]


            if self.gesture_count == 0:
                if abs(l_r) > 0.087 or abs(u_d) > 0.187:

                    if l_r < 0:
                        print("建议略微向右转头")
                    else:
                        print("建议略微向右转头")

                    if u_d < 0:
                        print("建议略微抬头")
                    else:
                        print("建议略微低头")

                else:
                    self.bg_list.append(image)
                    self.gesture_count += 1
            if self.gesture_count == 1:
                if u_d > -0.35:

                    print("请缓慢低头")

                else:
                    self.bg_list.append(image)
                    self.gesture_count += 1


            if self.gesture_count == 2:
                if l_r < 0.44:
                    print("请缓慢向右转头")

                else:
                    self.bg_list.append(image)
                    self.gesture_count += 1


            if self.gesture_count == 3:
                if l_r > -0.44:
                    print(l_r)
                    print("请缓慢向左转头")

                else:
                    self.bg_list.append(image)
                    self.gesture_count += 1

            print(self.gesture_count)
            return self.gesture_count

        except:

            print("-----no face")
            return self.gesture_count


    def create_face_vector(self,image=None, user_name='user_name_test', real_name='real_name_test'):

        gesture_count = self.live_test(image)
        if gesture_count <= 3:
            return False

        person = self.extension_img(self.bg_list)
        if person is None:
            return False

        Person.create_table()

        Person.delete().where(Person.user_name == user_name).execute()

        p_id = Person.insert({

            'user_name': user_name,
            'real_name': real_name,
            'face_data': json.dumps(person.face_data),
            'euc_dists': json.dumps(person.euc_dists),
            'cos_dists': json.dumps(person.cos_dists),

        }).execute()

        classifier_model.classify()
        print('数据处理完毕,next steps...')
        return True