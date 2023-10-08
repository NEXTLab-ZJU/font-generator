import os
import sys
import json
import cv2
from skimage import io,measure
from contour_tools import rasterize_cubic,contour2matrix
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
from copy import deepcopy

def cal_cnm(img):
    m = measure.moments(img)
    centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    mu = measure.moments_central(img, centroid)
    smu = measure.moments_normalized(mu)
    moments = []
    for j in range(16):
        nu = smu[j//4,j%4]
        if np.isnan(nu):
            continue
        else:
            moments.append(nu)
    return moments

def get_bbox(img):
    img = Image.fromarray(img)
    bbox = img.getbbox()
    return bbox

def refine_img(img,canvas_size,tar_size=None):
    if not tar_size:
        tar_size=canvas_size
    img = Image.fromarray(img)
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size - 1, r + 5)
    d = min(canvas_size - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  
    pad_len = int(abs(width - height) / 2)  

    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)

    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #
    img = img.squeeze(0)  # 
    img = transforms.ToPILImage()(img)
    img = img.resize((tar_size, tar_size), Image.ANTIALIAS)
    return np.array(img),bbox


def resize_contour(contours,bbox,canvas_size):
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size - 1, r + 5)
    d = min(canvas_size - 1, d + 5)

    f = max((d-u),(r-l))
    for i, c in enumerate(contours):
        for j, _ in enumerate(c):
            contours[i][j]['y'] -= (u+d)/2
            contours[i][j]['x'] -= (l+r)/2

    for i, c in enumerate(contours):
        for j, _ in enumerate(c):
            for key in contours[i][j].keys():
                if key=='on':
                    continue
                else:
                    contours[i][j][key] *=canvas_size/f
                    
    for i, c in enumerate(contours):
        for j, _ in enumerate(c):
            contours[i][j]['y'] += canvas_size/2
            contours[i][j]['x'] += canvas_size/2
    return contours



class Calculate_Moment(object):
    def __init__(self):
        self.tar_upm = 512
        self.is_finished = False
        self.progress = 0
        self.stroke_cnm = []
        self.data = {}

    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress

    def set_json_dir(self,json_dir):
        self.json_dir = json_dir
    
    def set_save_dir(self,save_dir):
        self.save_dir = save_dir
        
    def set_stroke_save_dir(self,stroke_save_dir):
        self.stroke_save_dir = stroke_save_dir
        
    def do(self,json_dir = None,save_dir = None,stroke_save_dir = None):
        if json_dir == None:
            json_dir = self.json_dir
        if save_dir == None:
            save_dir = self.save_dir
        if stroke_save_dir == None:
            stroke_save_dir = self.stroke_save_dir
            
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for root, dirs, files in os.walk(json_dir):
            for progress,file in enumerate(tqdm(files)):
                self.progress = (progress+1)/len(files)

                if file[0] == ".":
                    continue

                path = os.path.join(root, file)
                f = open(path, "r", encoding="utf-8") 
                content = json.load(f)

                upm = content["upm"]
                baseline = content["baseline"]
                infos = content["info"]

                contours = list(map(lambda x: x["contour"], infos))
                types_n_index = list(map(lambda x: {'type':x['type'],'id':x['id']}, infos))

                all_cnm = []
                normalized_img_path = []
                bboxs = []
                matrixs = []
                pixel_counts = []

                for i, c in enumerate(contours):
                    for j, _ in enumerate(c):
                        contours[i][j]['y'] += baseline
                        for key in contours[i][j].keys():
                            if key=='on':
                                continue
                            else:
                                contours[i][j][key] = contours[i][j][key] * self.tar_upm / upm

                for i, c in enumerate(contours):
                    for j, _ in enumerate(c):
                        contours[i][j]['y'] = self.tar_upm - contours[i][j]['y']
                        if 'delta_y_min' in contours[i][j].keys():
                            contours[i][j]['delta_y_min'] = -contours[i][j]['delta_y_min']
                            contours[i][j]['delta_y_max'] = -contours[i][j]['delta_y_max']

                total_img = rasterize_cubic(contours,self.tar_upm,(self.tar_upm,self.tar_upm))

                _,bbox = refine_img(total_img,self.tar_upm)
                contours = resize_contour(contours,bbox,self.tar_upm)

                
                for i, c in enumerate(contours):
                    img = rasterize_cubic([c], self.tar_upm, (self.tar_upm, self.tar_upm))
                    _, bi_img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
                    pixel_counts.append(np.sum(bi_img/255))
                    bbox = get_bbox(bi_img)

                    nor_img,_ = refine_img(deepcopy(img),self.tar_upm,128)
                    _, img = cv2.threshold(nor_img, 128, 255, cv2.THRESH_BINARY)
                    moments = cal_cnm(img)

                    all_cnm.append(moments)
                    l, u, r, d = bbox
                    bboxs.append( {'x_min':l,'x_max':r,'y_min':u,'y_max':d})
                    matrixs.append(contour2matrix(c))

                    self.stroke_cnm.append(moments)

                    type = types_n_index[i]['type']
                    type = str(type)
                    if not os.path.isdir(os.path.join(stroke_save_dir,type)):
                        os.makedirs(os.path.join(stroke_save_dir,type))
                    io.imsave(os.path.join(stroke_save_dir,type,file[0]+str(types_n_index[i]['id'])+'.png'),nor_img)
                    normalized_img_path.append(os.path.join(stroke_save_dir,type,file[0]+str(types_n_index[i]['id'])+'.png'))
                    
                for i, info in enumerate(infos):
                    info["cnm"] = all_cnm[i]
                    info["type"] = info["type"]
                    info["path"] = normalized_img_path[i]
                    info['bbox'] = bboxs[i]
                    info['matrix'] = matrixs[i]
                    info['pixel'] = pixel_counts[i]
                    del info['contour']

                self.data[file[0]] = content

        data_cnm = np.array(self.stroke_cnm)
        self.mean = np.mean(data_cnm,axis=0)
        self.std = np.std(data_cnm,axis=0)
        _,dim=data_cnm.shape

        for char,content in self.data.items():
            infos = content["info"]
            for info in infos:
                for i in range(dim):
                    info['cnm'][i] -= self.mean[i]
                    info['cnm'][i] /= self.std[i]

            result_path = os.path.join(save_dir, char+'.json')
            with open(result_path,"w",encoding="utf-8") as f:
                json.dump(content,f, indent=4)

        

        self.is_finished = True
        self.progress = 1
        return save_dir,self.mean,self.std