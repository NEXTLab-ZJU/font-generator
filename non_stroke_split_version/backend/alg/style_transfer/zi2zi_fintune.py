# -*- coding: utf-8 -*-

import os
import glob
import json
import os
import pickle
import random
from tqdm import tqdm
import sys

from zi2zi.font2img import *
from zi2zi.train import *
    
'''
zi2zi训练+finetune
数据整理
训练
fintune
图片名汉字_png
'''

def pickle_examples_with_split_ratio(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p, label in tqdm(paths):
                label = int(label)
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


def pickle_examples_with_file_name(paths, obj_path):
    with open(obj_path, 'wb') as fa:
        for p, label in tqdm(paths):
            label = int(label)
            with open(p, 'rb') as f:
                img_bytes = f.read()
                example = (label, img_bytes)
                pickle.dump(example, fa)


class Zi2ziFont(object):
    def __init__(self,pic_path,device,epoch=300,base_zi2zi_path = './zi2zi',base_zi2zi_result_path='./zi2zi_data'):
        self.base_zi2zi_path = base_zi2zi_path
        self.base_zi2zi_data_path = base_zi2zi_result_path
        self.picPath = pic_path
        self.basePicPath = base_zi2zi_path + '/base_train_data/'
        self.testPicpath = base_zi2zi_path + '/test_pic/'
        self.inferPicPath = base_zi2zi_result_path + '/infer_data/'
        #train pos stage=2 -> fintune
        self.stage = 1 
        #target font
        self.src_font = base_zi2zi_path + '/data/font_src/hywh.ttf'
        self.char_path= base_zi2zi_path + '/charset/gb2312.txt'
        self.charlist= list(open(self.char_path, encoding='utf-8').readline().strip())
        self.train_char = []
        # self.train_data = './Zi2zi/experiment/data'
        self.train_data = base_zi2zi_result_path + '/data'
        self.fintune_data = base_zi2zi_result_path + '/finetune/data'
        self.ckpt_dir = base_zi2zi_result_path + '/train_result/'
        # self.fintune_dir = base_zi2zi_result_path + '/fintune_result/'
        self.epoch = epoch
        self.label = 6
        self.canvas_size =256
        self.char_size = 256
        
        self.zi2zi_main = zi2ziMain(self.epoch,device)

    def set_result_path(self,dataPath):
        self.base_zi2zi_data_path = dataPath
        self.inferPicPath = self.base_zi2zi_data_path + '/infer_data/'
        if not os.path.exists(self.inferPicPath):
            os.makedirs(self.inferPicPath)
        self.train_data = self.base_zi2zi_data_path + '/data'

        if not os.path.exists(self.train_data):
            os.makedirs(self.train_data)
            
        self.fintune_data = self.base_zi2zi_data_path + '/finetune/data'

        if not os.path.exists(self.fintune_data):
            os.makedirs(self.fintune_data)
        self.ckpt_dir = self.base_zi2zi_data_path + '/train_result/'

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        # self.fintune_dir = self.base_zi2zi_data_path + '/fintune_result/'
        
        # if not os.path.exists(self.fintune_dir):
        #     os.makedirs(self.fintune_dir)
    
    def set_picpath(self,picPath):
        self.picPath = picPath
        
    def data_generate(self):
        print(self.picPath,self.testPicpath,self.src_font)
        self.train_char = font2img(self.picPath,self.testPicpath,self.src_font)
        total_file_list = sorted(
        glob.glob(os.path.join(self.basePicPath, "*.jpg"))+
        glob.glob(os.path.join(self.basePicPath, "*.png"))+
        glob.glob(os.path.join(self.testPicpath, "*.jpg"))+
        glob.glob(os.path.join(self.testPicpath, "*.png")))
        train_path = os.path.join(self.train_data, "train.obj")
        val_path = os.path.join(self.train_data, "val.obj")
        cur_file_list = []
        for file_name in total_file_list:
            cur_file_list.append((file_name, self.label))
        pickle_examples_with_split_ratio(
            cur_file_list,
            train_path=train_path,
            val_path=val_path,
            train_val_split=0.1
        )
    
    def train(self):
        self.data_generate()
        self.zi2zi_main.train(0.001,self.ckpt_dir,self.train_data)
    
    def fintune(self):
        self.train_char = font2img(self.picPath,self.testPicpath,self.src_font)
        total_file_list = sorted(
        glob.glob(os.path.join(self.testPicpath, "*.jpg"))+
        glob.glob(os.path.join(self.testPicpath, "*.png")))
        train_path = os.path.join(self.fintune_data, "train.obj")
        val_path = os.path.join(self.fintune_data, "val.obj")
        cur_file_list = []
        for file_name in total_file_list:
            cur_file_list.append((file_name, self.label))
        pickle_examples_with_split_ratio(
            cur_file_list,
            train_path=train_path,
            val_path=val_path,
            train_val_split=0.1
        )
        self.zi2zi_main.train(0.0001,self.ckpt_dir,self.fintune_data,1)
    
    def main_train(self):
        self.train()
        self.stage =2
        self.fintune()
    
    def getTrainState(self):
        return self.zi2zi_main.getTrainState()
    def getInferState(self):
        return self.zi2zi_main.getInferState()
    def getInferResult(self):
        return self.inferPicPath
    def getCkptDir(self):
        return self.ckpt_dir
    def infer(self,src_txt,ckpt_dir):
        epoch = self.epoch
        infer_dir = self.base_zi2zi_data_path + '/infer_data/'
        print(src_txt)
        src = ''.join(src_txt)
        
        font = ImageFont.truetype(self.src_font, size=self.char_size)

        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char_new(ch, font, self.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]

        label_list = [self.label for _ in src]
        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)
        self.zi2zi_main.infer(img_list,label_list,ckpt_dir,infer_dir,src,epoch)



        

if __name__ == "__main__":
    '''
    # Please input the directory containing the training data.
    zi2zi = Zi2ziFont('./Zi2zi/100/',1)# The first parameter is the path to the first 100 images, and the second parameter is the number of epochs (although there is a default value, you can remove it when actually using it).
    Requirement: The images should be named using Chinese characters, for example, "我.jpg" or "我.png".
    
    # The model training process consists of two steps: training and fine-tuning.
    zi2zi.main_train()
    
    print(zi2zi.getTrainState())
    
    # To obtain the checkpoint (ckpt) file path:
    result = zi2zi.getCkptDir() #ckptdir

    zi2zi.infer('我们是天下第一',zi2zi.ckpt_dir,1)# The first parameter is the character set, the second parameter is the checkpoint directory (ckptdir), and the third parameter is 1 for the number of epochs (used for testing, remove it for actual usage).
    
    print(zi2zi.getInferResult())
    
    print(zi2zi.getInferState())
    '''
    
