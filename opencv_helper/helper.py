# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show


def histogram(gray):
    """show histogram for a gray image.
    """
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(gray.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend( ('cdf', 'histogram'), loc = 'upper left')
    plt.show()


def match_more(img_orig, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    """match original image with a template,
        return all matches that above a threshold.
        The higher a threshold is, the better it matches
    """
    img = img_orig.copy()
    h, w = template.shape
    res = cv2.matchTemplate(img, template, method)
    loc = np.where( res > threshold )
    for pt in zip(*loc[:]):
        top_left = (pt[1], pt[0])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.imshow(img, cmap='gray')
    plt.show()


def match_template(img_orig, template, method):
    """same as the above, but returun the best matched area.
    """
    img = img_orig.copy()
    h, w = template.shape
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # print(top_left, bottom_right)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.imshow(img, cmap='gray')
    plt.show()


def watershed(img_orig):
    """watershed requires a 3-channel picture!
    """
    img_cp = img_orig.copy()
    img = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)

    # use threshold to filter out chips-under.
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones( (3,3), np.uint8)
    # dilate image to find sure background.
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    # erode image to find sure foreground.
    sure_fg = cv2.erode(thresh, kernel, iterations=3)
    # difference of the two is unknown.
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # mark pixels. unknown is always 0.
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img_cp, markers)
    
    img_cp[markers == -1] = [255, 0, 0]
    # return cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    return img_cp


def SIFT_obj_identify(img1, img2):
    """
    Locate img1 from img2 using SIFT feature extraction and KNN matching alg.
    Example taken from: https://docs.opencv.org/3.3.1/da/df5/tutorial_py_sift_intro.html
    SIFT: scale-invariant feature transform.
    :param img1: query image in gray scale
    :param img2: train image in gray scale
    :return: img3, a combination of img1 and img2
    """
    img1_cp = img1.copy()
    img2_cp = img2.copy()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_cp, None)
    kp2, des2 = sift.detectAndCompute(img2_cp, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1_cp.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        print(dst)

        img2_cp = cv2.polylines(img2_cp, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatches(img1_cp, kp1, img2_cp, kp2, good, None, **draw_params)
    return img3, img2_cp
