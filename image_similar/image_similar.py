import cv2
from image_similar import ssim
import os
import json

def compare_images(img1_path,img2_path):

    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))
        # print(vifp.vifp_mscale(img1,img2))
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        result = ssim.ssim_exact(img1, img2)
        result = '{}'.format(result)
        dic = {'AppType': '2', 'LocalPath': result}
        resultstr = json.dumps(dic)

    except:
        dic = {'AppType': '2', 'error':'图片获取出错'}
        resultstr = json.dumps(dic)
        return resultstr

    return resultstr






