import os
import string

import torchvision
from torchvision import  transforms
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

from torch.utils.data import Dataset, ConcatDataset, Subset, TensorDataset
from recog.model import Model
from recog.utils import CTCLabelConverter, AttnLabelConverter
import argparse
from PIL import Image
from recog.dataset  import ImgDataset, AlignCollate
import time
import cv2
import numpy as np


class OCR:

    def __init__(self, args):
        print("OCR is initializing ...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('self.device ',self.device)

        """ vocab / character number configuration """
        # if args['sensitive:
        #     args['character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        cudnn.benchmark = True
        cudnn.deterministic = True
        args['num_gpu'] = torch.cuda.device_count()

        self.saved_model = args['saved_model']
        #number of data loading workers
        self.workers = args['workers']
        #input batch size
        self.batch_size = args['batch_size']
        # self.saved_model = args['saved_model
        #maximum-label-length
        self.batch_max_length = args['batch_max_length']
        #the height of the input image
        self.imgH = args['imgH']
        #the width of the input image
        self.imgW = args['imgW']
        #use rgb input
        self.rgb = args['rgb']
        #character label
        self.character = args['character']
        # self.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựỷỹỳýỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬđĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰỶỸỲÝỴ '
        #for sensitive character mode
        self.sensitive = args['sensitive']
        #whether to keep ratio then pad for image resize
        self.PAD = args['PAD']
        #Transformation stage. None|TPS
        self.Transformation = args['Transformation']
        #FeatureExtraction stage. VGG|RCNN|ResNet
        self.FeatureExtraction = args['FeatureExtraction']
        #SequenceModeling stage. None|BiLSTM
        self.SequenceModeling = args['SequenceModeling']
        #Prediction stage. CTC|Attn
        self.Prediction = args['Prediction']
        #number of fiducial points of TPS-STN
        self.num_fiducial = args['num_fiducial']
        #the number of input channel of Feature extractor
        self.input_channel = args['input_channel']
        #the number of output channel of Feature extractor
        self.output_channel = args['output_channel']
        #the size of the LSTM hidden state
        self.hidden_size = args['hidden_size']

        self.num_gpu = args['num_gpu']



        if 'CTC' in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)
        args['num_class'] = self.num_class
        
        if self.rgb:
            self.input_channel = 3
            args['input_channel'] = 3
        self.model = Model(args)

        print('model input parameters', self.imgH, self.imgW, self.num_fiducial, self.input_channel, self.output_channel,
              self.hidden_size, self.num_class, self.batch_max_length, self.Transformation, self.FeatureExtraction,
              self.SequenceModeling, self.Prediction, flush=True)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % self.saved_model, flush = True)
        self.model.load_state_dict(torch.load(self.saved_model, map_location=self.device))
        self.model.eval()

        self.allignCollate = AlignCollate(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)


    def predict(self, dataset):
        # print('predict')
        if dataset is not None:
            results = []
            data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=int(self.workers),
                    collate_fn=self.allignCollate, pin_memory=True)
            # self.model.eval()
            with torch.no_grad():
                for image_tensors, index in data_loader:
                    # print("image_tensors ", image_tensors)
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(self.device)
                     # For max length prediction
                    length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(self.device)
                    text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(self.device)

                    if 'CTC' in self.Prediction:
                        preds = self.model(image, text_for_pred)

                        #Select max probabilty (greedy decoding) then decode index to character
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.view(-1)
                        preds_str = self.converter.decode(preds_index.data, preds_size.data)

                    else:
                        # print('Attn')
                        preds = self.model(image, text_for_pred, is_train=False)
                        # select max probabilty (greedy decoding) then decode index to character
                        _, preds_index = preds.max(2)
                        preds_str = self.converter.decode(preds_index, length_for_pred)

                        if preds_str == []:
                            preds_str = [[" "]]

                    log = open(f'./log_demo_result.txt', 'a', encoding="utf-8")
                    dashed_line = '-' * 80
                    head = f'{"predicted_labels":25s}\tconfidence score'

                    # print(f'{dashed_line}\n{head}\n{dashed_line}')
                    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    
                    for pred, pred_max_prob, pred_index in zip(preds_str, preds_max_prob, index):
                   
                        if 'Attn' in self.Prediction:
                            pred_EOS = pred.find('[s]')
                            if pred_EOS == 0:
                                pred = " "
                                pred_max_prob = torch.tensor([1.0]) #pred_max_prob[:pred_EOS]
                            else:
                                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                pred_max_prob = pred_max_prob[:pred_EOS]
                        # calculate confidence score (= multiply of pred_max_prob)
                        try:
                           confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                        except Exception as e:
                            print(e)
                            confidence_score = torch.tensor([1.0])

                        log.write(f'{pred:25s}\t{confidence_score:0.4f}\n')
                        # if confidence_score < torch.tensor(0.3,device=self.device):
                        #     pred = " "
                        if len(pred) == 1 and confidence_score < torch.tensor(0.75,device=self.device):
                            pred = " "
                        results.append([pred, confidence_score, pred_index])

                    log.close()
            return  results

        else:
            return ""

if __name__ == '__main__':

    args = {}
    args['workers'] = int(os.environ.get('workers', 4))
    args['batch_size'] = int(os.environ.get('batch_size', 192))
    args['saved_model'] = (os.environ.get('saved_model', 'recog/model/best_accuracy.pth'))
    args['batch_max_length'] = int(os.environ.get('batch_max_length', 25))
    args['imgH'] = int(os.environ.get('imgH', 32))
    args['imgW'] = int(os.environ.get('imgW', 100))
    args['rgb'] = (os.environ.get('rgb', False))
    args['character'] = (os.environ.get('character',
                                        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựỷỹỳýỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬđĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰỶỸỲÝỴ '))
    args['sensitive'] = (os.environ.get('sensitive', True))
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

    dir_path = "../static/out/"
    imgs = []
    labels = []
    # trans = transforms.ToTensor()
    # iterate through the names of contents of the folder
    for image_path in sorted(os.listdir(dir_path)):
        input_path = os.path.join(dir_path, image_path)
        if ocr.rgb:
            img = Image.open(input_path).convert('RGB')
        else:
            img = Image.open(input_path).convert('L')
        imgs.append(img)
        labels.append(image_path)

    data_test = ImgDataset(imgs,labels)
    start = time.time()
    results = ocr.predict(data_test)
    print("Recognize ",str(len(imgs)) ," words  took {} seconds.". format(time.time() - start))
    # print(results)

    for pred, score, lable in results:
        print("====",pred,"====",score, "=====", lable)
