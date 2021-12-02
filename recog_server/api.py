# Import the required modules
import argparse

import base64
import cv2
import io
import numpy as np
from PIL import Image
from recog.OCR import OCR
from recog.dataset  import ImgDataset, AlignCollate
import time
import json
import os, shutil
#parser = argparse.ArgumentParser()
# parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
args = {}
args['workers'] = int(os.environ.get('workers', 4))
args['batch_size'] = int(os.environ.get('batch_size', 192))
args['saved_model'] = (os.environ.get('saved_model', 'recog/model/best_accuracy.pth'))
args['batch_max_length'] = int(os.environ.get('batch_max_length',25 ))
args['imgH'] = int(os.environ.get('imgH', 32))
args['imgW'] = int(os.environ.get('imgW', 100))
args['rgb'] = (os.environ.get('rgb',False ))
args['character'] = (os.environ.get('character', '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựỷỹỳýỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬđĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỐỖỘÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ '))
args['sensitive'] = (os.environ.get('sensitive',True ))
args['PAD'] = (os.environ.get('PAD', False))
args['Transformation'] = (os.environ.get('Transformation', 'TPS'))
args['FeatureExtraction'] = (os.environ.get('FeatureExtraction', 'ResNet'))
args['SequenceModeling'] = (os.environ.get('SequenceModeling', 'BiLSTM'))
args['Prediction'] = (os.environ.get('Prediction', 'Attn'))
args['num_fiducial'] = int(os.environ.get('num_fiducial', 20))
args['input_channel'] = int(os.environ.get('input_channel', 1))
args['output_channel'] = int(os.environ.get('output_channel', 512))
args['hidden_size'] = int(os.environ.get('hidden_size', 256))


ocr = OCR(args)

def is_grey_scale(img):
    img = img.convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True

def batch_process(data):
    try:
        base64_image = data['image']
        file_like = io.BytesIO(base64.b64decode(base64_image))
                
        input_image = Image.open(file_like)
        if not is_grey_scale(input_image):
            input_image = input_image.convert("L")

        text_bounds = data['text_bounds'][:-1].split(',')
        int_text_bounds = [int(numeric_string) for numeric_string in text_bounds]
        re_order = np.asarray(int_text_bounds).reshape(-1, 4)

        segs = []
        labels = []

        pos_index = -1

        if os.path.isdir("static/out/") == False:
            os.mkdir("static/out/")
        else:
            for filename in os.listdir('static/out'):
                file_path = os.path.join('static/out', filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        for i, i_row in enumerate(re_order):

            text_crop = input_image.crop((i_row[0], i_row[1], i_row[2], i_row[3]))
            if text_crop is None :
                continue
            w_crop, h_crop = text_crop.size    
            text_crop.save("static/out/"+str(i)+".png")
            segs.append(text_crop)
            labels.append(str(pos_index))

        data_test = ImgDataset(segs,labels)
        results = ocr.predict(data_test)
        
        preds = []
        scores = []
        for pred, score, label in results:
            preds.append(pred)
            scores.append(score.cpu().data.numpy().astype(float).tolist())
        index_score = -1
        output_str = ""
        list_score = []
        for i_row in preds:
            index_score += 1
            if output_str == "":
                output_str = output_str + preds[index_score]
            else:
                output_str = output_str + " " + preds[index_score]
            list_score.append(scores[index_score])
        json_data = {"result_code": 200, "id_check":"","id_logic":"","id_logic_message":"","id_type":"","idconf":"[]",
                    "server_name":"","server_ver":"1.0", "words":"", "scores":""}

        json_data['words'] = output_str
        json_data['scores'] = list_score

        if (len(results) > 0):
            return json_data
        else:
            return {"result_code":500, "id_check":"","id_logic":"","id_logic_message":"","id_type":"","idconf":"[]",
                    "server_name":"","server_ver":"1.0", "words":"", "scores":-1}
    except Exception as e:
        print(e)
        return {"result_code":500, "id_check":"","id_logic":"","id_logic_message":"","id_type":"","idconf":"[]",
                    "server_name":"","server_ver":"1.0", "words":"", "scores":-1}
