import numpy as np
import cv2
import os
import time
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

top_3_list = []

def CrossCorrelation(im1, im2):
    global top_3_list

    # print(im2)

    img1 = im1
    img2 = cv2.imread(im2)

    result_list = []

    width1 = img1.shape[1]
    width2 = img2.shape[1]

    for i in range(width2-width1):
        cropImg = img2[0:img2.shape[0], i:i+width1]

        # 图片2的标准差
        # print(np.std(img2))
        # 相关系数，这里使用的是有偏估计
        coi = np.mean(np.multiply((img1-np.mean(img1)),(cropImg-np.mean(cropImg))))/(np.std(img1)*np.std(cropImg))
        if top_3_list == None or len(top_3_list) < 3:
            list_item = [coi, im2, i]
            top_3_list.append(list_item)
            top_3_list = sorted(top_3_list, key=(lambda x: x[0]))
        elif coi > top_3_list[0][0]:
            del top_3_list[0]
            list_item = [coi, im2, i]
            top_3_list.append(list_item)
            top_3_list = sorted(top_3_list, key=(lambda x: x[0]))
        # result_list.append(coi)


def normxcorr2(template, image, mode="same"):
    global top_3_list

    template_shape = template.shape
    image_name = image

    image = cv2.imread(image, 0)

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    if top_3_list == None or len(top_3_list) < 3:
        index = np.where(out == np.max(out))
        list_item = [np.max(out), image_name, int(index[1] - template_shape[1] / 2)]
        top_3_list.append(list_item)
        top_3_list = sorted(top_3_list, key=(lambda x: x[0]))
    elif np.max(out) > top_3_list[0][0]:
        del top_3_list[0]
        index = np.where(out == np.max(out))
        list_item = [np.max(out), image_name, int(index[1] - template_shape[1] / 2)]
        top_3_list.append(list_item)
        top_3_list = sorted(top_3_list, key=(lambda x: x[0]))

    # if np.max(out) > max_value:
    #     max_value = np.max(out)
    #     image_path = image_name
    #     index = np.where(out == np.max(out))
    #     # print(index)
    #     # print(template_shape)
    #     shift = int(index[1] - template_shape[1] / 2)


def TemplateMatchingUsingCorrelation(img, folder):
    global top_3_list

    start = time.time()

    image = cv2.imread(img)
    sp = image.shape
    sz1 = sp[0] # height
    sz2 = sp[1] # width
    a = 0
    b = sz1
    c = int(sz2/2-sz2/6)
    d = int(sz2/2+sz2/6)
    cropImg = image[a:b, c:d]
    # plt.imshow(cropImg)
    # plt.show()
    files = os.listdir(folder)

    # Cross-Correlation
    for i in range(1, len(files)+1):
        CrossCorrelation(cropImg, folder+"/"+str(i)+".jpg")

    end = time.time()

    print("Running time: " + str(end-start))
    # print(top_3_list)
    for i in top_3_list:
        print(i)

def TemplateMatchingUsingConvolution(img, folder):
    global max_value, image_path, shift


    # Grayscale
    # gray = cv2.imread(img, 0)

    # R, G, B channel
    # image = cv2.imread(img)
    # b = image[:,:,0]
    # g = image[:,:,1]
    # r = image[:,:,2]

    # cv2.imshow("b", b)
    # cv2.imshow("g", g)
    # cv2.imshow("r", r)
    # cv2.waitKey()
    # cv2.destroyWindow()


    for i in range(4):

        start = time.time()

        if i == 0:
            print("GrayScale")
            image = cv2.imread(img, 0)
        elif i == 1:
            print("B")
            image = cv2.imread(img)
            image = image[:, :, 0]
        elif i == 2:
            print("G")
            image = cv2.imread(img)
            image = image[:, :, 1]
        elif i == 3:
            print("R")
            image = cv2.imread(img)
            image = image[:, :, 2]
        sp = image.shape
        # print(sp)
        sz1 = sp[0]  # height
        sz2 = sp[1]  # width
        a = 0
        b = sz1
        c = int(sz2 / 2 - sz2 / 6)
        d = int(sz2 / 2 + sz2 / 6)
        cropImg = image[a:b, c:d]
        # cv2.imshow("cropImg", cropImg)
        # cv2.waitKey()
        # cv2.destroyWindow()
        # print(cropImg.shape)
        # plt.imshow(cropImg)
        # plt.show()
        files = os.listdir(folder)

        # result = normxcorr2(cropImg, folder + "/1.jpg")

        for i in range(1, len(files) + 1):
            normxcorr2(cropImg, folder + "/" + str(i) + ".jpg")

        end = time.time()

        print("Running time: " + str(end - start))
        # print(top_3_list)
        for i in top_3_list:
            print(i)



if __name__ == '__main__':
    # TemplateMatchingUsingCorrelation("/home/sheldon/ImageSet/1/sub_level_2/1.jpg", "/home/sheldon/ImageSet/6/sub_level_2")
    TemplateMatchingUsingConvolution("/home/sheldon/ImageSet/1/sub_level_6/10.jpg", "/home/sheldon/ImageSet/1/sub_level_2")