import random
import time

import cv2
import pyscreeze
from PIL import Image
import pyautogui
import numpy as np
# from paddleocr import PaddleOCR
# from paddleocr.tools.infer.utility import draw_ocr

button_path = "./resource/img/button.png"
button_img = Image.open('./resource/img/button.png')
length_img = int((button_img.size[0] - 2) / 10)
button_img_size = button_img.size
buttons = []
for i in range(10):
    # left up right down
    img_temp = button_img.crop(
        (i * length_img + 1, 1, i * length_img + length_img + 1, button_img_size[1] - 1))
    buttons.append(img_temp)
    # img_temp.show()
    # print(img_temp.size)
    # input("waiting")


def get_random_score(score, weight):
    result = random.choices(score, weights=weight)[0]
    print(f"随机结果:{result}\n")
    return result


def get_target():
    """
    获取匹配的元素的位置
    :return: 列表中包含每个元素的坐标范围{p1:point1, p2:point2} point1:(x,y)
    """
    # 需要匹配的模板
    target = cv2.imread(button_path, cv2.IMREAD_GRAYSCALE)
    # target = cv2.cvtColor(np.asarray(buttons[get_random_score(scores) - 1]), cv2.COLOR_RGB2GRAY)
    # cv2.imshow('target', target)
    # 截图
    screenshot = pyscreeze.screenshot()
    # 将截图转化为cv2的灰度图
    temp = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2GRAY)
    # 获取模板的尺寸
    h, w = target.shape[:2]
    # print(f"匹配模板尺寸:{h}x{w}\n")
    # cv2模板匹配
    ret = cv2.matchTemplate(temp, target, cv2.TM_CCOEFF_NORMED)
    # 获取匹配相似>0.8的值
    index = np.where(ret > 0.75)
    # print(f"index:{index}")
    target_points = []
    # 用于展示的图片
    draw_img = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)
    for i in zip(*index[::-1]):
        # print(f"i:{i}")
        # 画出每个匹配的元素（若需要展示）
        rect = cv2.rectangle(draw_img, i, (i[0] + w, i[1] + h), (0, 0, 255), 1)
        target_points.append({"p1": i, "p2": (i[0] + w, i[1] + h)})
        # print(f"匹配位置:{i}x{i[0] + w, i[1] + h}")
    # 展示匹配
    # cv2.imshow('rect', draw_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return target_points


def score(target_score: tuple = (8, 9, 10), weight: tuple = (1, 7, 2)):
    cache_img = pyscreeze.screenshot()
    while True:
        target = get_target()
        for t in target:
            target_length = t["p2"][0] - t["p1"][0]
            point = (
                (target_length / 10) * int(get_random_score(target_score, weight)) + t["p1"][0] - (target_length / 10 / 2),
                (t["p2"][1] - t["p1"][1]) / 2 + t["p1"][1])
            print(f"目标点:{point[0]}x{point[1]}")
            pyautogui.click(point)
            time.sleep(0.1)
        pyautogui.scroll(-150)
        time.sleep(0.1)
        imgs = pyscreeze.screenshot()
        if cache_img == imgs:
            break
        else:
            cache_img = imgs


# def orc_img_text(path="", saveImg=False, printResult=False):
#     """
#     图像文字识别
#     :param path: 图片路径
#     :param saveImg: 是否把结果保存为图片
#     :param printResult: 是否打印出识别结果
#     :return: result, img_name
#     """
#     if path == "":
#         image = pyautogui.screenshot()
#         image = np.array(image)
#     else:
#         image = Image.open(path).convert('RGB')
#
#     ocr = PaddleOCR(use_angle_cls=True, lang='ch')
#
#     result = ocr.ocr(image, cls=True)
#
#     if printResult is True:
#         for line in result:
#             for word in line:
#                 print(word)
#
#     img_name = "ImgTextOCR-img-" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".jpg"
#     if saveImg is True:
#         boxes = [detection[0] for line in result for detection in line]
#         txts = [detection[1][0] for line in result for detection in line]
#         scores = [detection[1][1] for line in result for detection in line]
#         im_show = draw_ocr(image, boxes, txts, scores)
#         im_show = Image.fromarray(im_show)
#         im_show.save(img_name)
#     return result, img_name


if __name__ == '__main__':
    time.sleep(1)
    score()
