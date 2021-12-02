# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # result directory

        unsortedBoxes = []
        for i, box in enumerate(boxes):
            poly1 = np.array(box).astype(np.int32).reshape((4, -1))
            poly = np.array([poly1[0,:], poly1[2,:]]).astype(np.int32).reshape((-1))
            unsortedBoxes.append(poly)
        SortedBox = PhanBlock(unsortedBoxes)
        # sortedBox = sorted(unsortedBoxes, key=lambda b:b[0], reverse=False)

        s = ''
        for i, box in enumerate(SortedBox):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + ','
            s += strResult

        return s
                # poly = poly.reshape(-1, 2)
                # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                # ptColor = (0, 255, 255)
                # if verticals is not None:
                #     if verticals[i]:
                #         ptColor = (255, 0, 0)

                # if texts is not None:
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     font_scale = 0.5
                #     cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                #     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

def PhanBlock(data):
    Lines = []
    LineToSort = []
    SortedBox = []

    for i, dulieu in enumerate(data):
        top = dulieu[1]
        bot = dulieu[3]

        if i == 0 or abs(Lines[i-1] - top) > height:
            Lines.append(top)
        else:
            Lines.append(Lines[i-1])

        height = bot - top

    for i, l in enumerate(Lines):
        if Lines[i] == Lines[i-1] or len(LineToSort) == 0:
            LineToSort.append(data[i])
        else:
            SortedLine = sorted(LineToSort, key=lambda b:b[0], reverse=False)
            SortedBox.extend(SortedLine)
            LineToSort = []
            LineToSort.append(data[i])
    
    return SortedBox
