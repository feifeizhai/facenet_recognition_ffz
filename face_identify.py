import cv2

from utils.db_manager import Person
from utils import image_processing
from utils import face_recognition

resize_width = 160
resize_height = 160


class face_identify():
    def __init__(self):
        self.face_detect = face_recognition.MTCNN()
        self.face_net = face_recognition.facenetEmbedding()
        self.classify = face_recognition.Classifier()

    def face_rect(self, image=None):

        bboxes, landmarks = self.face_detect.detect_face(image, fixed="height")

        return bboxes, landmarks

    def face_512_vector(self, face_images=None):


        face_images = image_processing.get_prewhiten_images(face_images)
        pred_emb = self.face_net.get_embedding(face_images)
        return pred_emb

    def CLAHES(self, imgs):

        new_imgs = []
        for img in imgs:
            new_img = self.CLAHE(img)
            new_imgs.append(new_img)

        return new_imgs

    def CLAHE(self, img, clipLimit=2.0,
              tileGridSize=(4, 4)):
        # img = img.astype(np.float32)

        clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                tileGridSize=tileGridSize)
        new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 限制对比度的自适应阈值均衡化
        new_image = clahe.apply(new_image)
        # new_image = cv2.bilateralFilter(new_image, 4, 75, 75)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)

        return new_image

    def face_recognition_image(self, image):

        bboxes, landmarks = self.face_rect(image)
        if len(bboxes) == 0:
            print("-----no face")
            return []

        print("-----image have {} faces".format(len(bboxes)))

        face_images = self.CLAHES(image_processing.get_bboxes_image(image, bboxes, landmarks, resize_height, resize_width))

        pred_embs = self.face_512_vector(face_images)

        pred_real_name, pred_user_name, pred_score = self.compare_embadding(pred_embs)

        # 在图像上绘制人脸边框和识别的结果
        show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_user_name, pred_score)]
        image = image_processing.show_image_bboxes_text("face_recognition", image, bboxes, show_info)
        cv2.imshow('123', image)
        cv2.waitKey(1)

        return pred_real_name, pred_user_name

    def compare_embadding(self, pred_emb, correcthold=0.8):

        predicts, predictions = self.classify.predict(pred_emb)
        pred_user_name = []
        pred_real_name = []
        pred_score = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            predict = predicts[i]
            person = Person.select().where(Person.user_name == predict)[0]
            real_name = person.real_name
            user_name = person.user_name

            correct = max(prediction)
            if correct < correcthold:
                real_name = 'unknow'
                user_name = 'unknow'

            pred_real_name.append(real_name)
            pred_user_name.append(user_name)
            pred_score.append(correct)

        return pred_real_name, pred_user_name, pred_score