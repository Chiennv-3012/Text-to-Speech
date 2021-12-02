import os

import torch
import torch.nn as nn
import re
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import cv2
from skimage import io
import numpy as np
from CRAFT import craft_utils
from CRAFT import imgproc
from CRAFT import file_utils

from CRAFT.craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text):
    # resize
    canvas_size = 1280 # image size for inference
    # mag_ratio  = 1.5 # image magnification ratio
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio = 1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def crop(inFolderImage, inFolderText):
    dirsImage = os.listdir(inFolderImage)
    dirsText = os.listdir(inFolderText)
    
    outFolder = "./resultImage/"
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for itemImage in dirsImage:
        fullpathImage = os.path.join(inFolderImage,itemImage)         
        pathResult = os.path.join(outFolder,itemImage)
        
        if os.path.isfile(fullpathImage):
            imageFileName, imageFileExtension = os.path.splitext(fullpathImage)
            imageName = imageFileName.split("/")
            for itemText in dirsText:
                fullpathText = os.path.join(inFolderText, itemText)
                if os.path.isfile(fullpathText):
                    textFileName, textFileExtension = os.path.splitext(fullpathText)
                    textname = textFileName.split("/")
                    if (textname[len(textname) - 1] == imageName[len(imageName) - 1]):
                        f = open(fullpathText, "r")
                        i = 1
                        for str in f:
                            im = Image.open(fullpathImage)
                            p,s = os.path.splitext(pathResult)
                            d = map(int, re.findall(r'\d+', str))
                            imCrop = im.crop((d))      
                            imCrop.save(p + 'Cropped' + '% s' %i + '.png', "PNG", quality=100)
                            i += 1
    return outFolder

def getContour(inImagefolder, fileName):
    image_list, _, _ = file_utils.get_files(inImagefolder)

    # load net
    net = CRAFT()     # initialize

    net.load_state_dict(copyStateDict(torch.load('CRAFT/weights/craft_mlt_25k.pth', map_location='cpu')))

    net.eval()

    resultS = ''
    # load data
    for k, image_path in enumerate(image_list):
        if (image_path == os.path.join(inImagefolder, fileName)):
            print("Test image {:s}".format(image_path), end='\r')
            image = imgproc.loadImage(image_path)
        
            text_threshold = 0.7 # text confidence threshold
            low_text = 0.4 # text low-bound score
            link_threshold = 0.4 # link confidence threshold

            bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text)

            resultS += file_utils.saveResult(image_path, image[:,:,::-1], polys)

    return resultS

