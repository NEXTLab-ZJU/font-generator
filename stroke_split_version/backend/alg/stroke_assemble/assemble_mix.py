import json,os
import cv2
import numpy as np
from skimage import io,measure
from contour_tools import rasterize_cubic,matrix2contour
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from torch import nn
import time
merge = {
    0:0,33:0,
    1:1,34:1,12:6,
    2:2,
    3:3,4:3,14:3,
    5:4,32:4,
    6:5,7:5,8:5,18:5,22:5,25:5,27:5,36:5,30:5,31:5,

    21:5,28:5,29:5,
    20:5,23:5,24:5,26:5,

    9:6,10:6,11:6,15:6,17:6,19:6,35:6,13:6,16:6,
    37:7
}
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
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 255
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)/255  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((tar_size, tar_size), Image.ANTIALIAS)
    return np.array(img),bbox

def compute_l1_loss(img1, img2):
    return np.average(np.abs(img1-img2))

def get_bounding_box(on_pts):
    bbox = np.array([np.min(on_pts, axis=0),np.max(on_pts, axis=0)])
    return {'x_min':bbox[0,0],'x_max':bbox[1,0],'y_min':bbox[0,1],'y_max':bbox[1,1]}

def new_cal_cnm(img,mean,std):
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
    assert len(moments)==len(mean)
    for i in range(len(moments)):
        moments[i] -= mean[i]
        moments[i] /= std[i]
    return moments

def new_rescale(ref,candidate):
    x_factor = (ref['bbox']['x_max']-ref['bbox']['x_min'])/(candidate['bbox']['x_max']-candidate['bbox']['x_min'])
    y_factor = (ref['bbox']['y_max']-ref['bbox']['y_min'])/(candidate['bbox']['y_max']-candidate['bbox']['y_min'])
    
    x_bias_old = (candidate['bbox']['x_max']+candidate['bbox']['x_min'])/2
    y_bias_old = (candidate['bbox']['y_max']+candidate['bbox']['y_min'])/2

    x_bias_new = (ref['bbox']['x_max']+ref['bbox']['x_min'])/2
    y_bias_new = (ref['bbox']['y_max']+ref['bbox']['y_min'])/2

    transfer_cord1 =np.array([[1,0,0],
                             [0,1,0],
                             [-x_bias_old,-y_bias_old,1]])
    
    transfer_cord2 =np.array([[x_factor,0,0],
                             [0,y_factor,0],
                             [x_bias_new,y_bias_new,1]])
    
    candidate['matrix']['cordinate'] = candidate['matrix']['cordinate']@transfer_cord1@transfer_cord2

    transfer_delta =np.array([[x_factor,0,0],
                             [0,y_factor,0],
                             [0,0,1]])
    
    if 'delta_min' in candidate['matrix'].keys():
        candidate['matrix']['delta_min'] = np.matmul(candidate['matrix']['delta_min'],transfer_delta)
    if 'delta_max' in candidate['matrix'].keys():
        candidate['matrix']['delta_max'] = np.matmul(candidate['matrix']['delta_max'],transfer_delta)

    candidate['contour'] = matrix2contour(candidate['matrix'])
    return candidate

class Stroke_Assemble(object):
    def __init__(self,upm,top_num):
        self.is_finished = False
        self.progress = 0
        self.upm=upm
        self.top_num=top_num

    def set_cnm_mean_std(self,cnm_mean,cnm_std):
        self.cnm_mean = cnm_mean
        self.cnm_std = cnm_std
        
    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress

    def set_split_result_dir(self,split_result_dir):
        self.split_result_dir = split_result_dir
    
    def set_stroke_data_dir(self,stroke_data_dir):
        self.stroke_data_dir = stroke_data_dir
    
    def set_json_save_dir(self,json_save_dir):
        self.json_save_dir = json_save_dir
    
    def load_stroke_dataset(self,stroke_data_dir):
        font_stroke_data = {}
        dataset = open(os.path.join(stroke_data_dir,'stroke_dataset.json'), "r", encoding="utf-8") 
        dataset = json.load(dataset)
        for type,values in dataset['vector_data'].items():
            merge_type = str(merge[int(type)])
            if merge_type not in font_stroke_data.keys():
                font_stroke_data[merge_type]=[]

            for stroke in values['contour']:
                font_stroke_data[merge_type].append(stroke)
        return font_stroke_data

    def count_min_pixel(self,font_stroke_data):
        min_counts = {}
        for stroke_type,strokes in font_stroke_data.items():
            min_count =self.upm*self.upm
            for stroke in tqdm(strokes):
                stroke['pixel']
                if stroke['pixel'] < min_count:
                    min_count = stroke['pixel']
            min_counts[stroke_type] = min_count
        
        return min_counts
        
    def extract_stroke_refs(self,file):
        refs = []

        for type in self.all_type:
            im = cv2.imread(os.path.join(self.split_result_dir,type,file), cv2.IMREAD_GRAYSCALE)
            if np.all(im==0):
                continue
            _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
            im = 255-im
            contours, h = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for i,cnt in enumerate(contours):
                if cv2.contourArea(cnt) < self.min_pixel[type]*0.7:
                    continue
                canvas = np.zeros((self.upm,self.upm))
                canvas = cv2.drawContours(canvas,[cnt],-1,255,-1)
                bbox = np.array([np.min(cnt, axis=0)[0],np.max(cnt, axis=0)[0]])
                bbox = {'x_min':bbox[0,0],'x_max':bbox[1,0],'y_min':bbox[0,1],'y_max':bbox[1,1]}
                normalized_img,_ = refine_img(deepcopy(canvas),self.upm,128)
                _, img = cv2.threshold(normalized_img, 128, 255, cv2.THRESH_BINARY)
                moments = new_cal_cnm(img,self.cnm_mean,self.cnm_std)
                refs.append({'type':type,'img':canvas,'cnm':moments,'n_img':normalized_img,'bbox':bbox})

        return refs

    def find_candidates(self,ref):
        candidates_l1 = []
        candidates_cnm = []

        for stroke in self.font_stroke_data[ref['type']]:
            stroke_normalized_img = io.imread(stroke['path'])
            dist_l1 = compute_l1_loss(stroke_normalized_img,ref['n_img'])
            dist_cnm = np.linalg.norm(np.array(ref['cnm'])-np.array(stroke['cnm']))
            candidates_l1.append((dist_l1,deepcopy(stroke)))
            candidates_cnm.append((dist_cnm,deepcopy(stroke)))

        candidates_l1.sort(key=lambda x:x[0])
        candidates_cnm.sort(key=lambda x:x[0])
        
        candidate = candidates_l1[:self.top_num] + candidates_cnm[:self.top_num]

        return candidate

    def overlap_filter(self,all_strokes):
        for c,all_stroke in enumerate(all_strokes):
            if c>=1:
                break
            for i in range(len(all_stroke)):
                if all_stroke[i]['drop']:
                    continue
                img1 = all_stroke[i]['bi_img']
                count1 = np.sum(img1)
                for j in range(len(all_stroke)):
                    if i==j or all_stroke[j]['drop']:
                        continue
                    img2 = all_stroke[j]['bi_img']
                    count2 = np.sum(img2)

                    res1 = deepcopy(img1)
                    res1[img2==1] = 0
                    res1_count = np.sum(res1)

                    res2 = deepcopy(img2)
                    res2[img1==1] = 0
                    res2_count = np.sum(res2)
                    if res1_count < 0.3*count1 or res2_count < 0.3*count2:

                        if all_stroke[i]['dist'] < all_stroke[j]['dist']:
                            for w in range(len(all_strokes)):
                                if i < len(all_strokes[w]):
                                    all_strokes[w][i]['drop']=True
                        else:
                            for w in range(len(all_strokes)):
                                if j < len(all_strokes[w]):
                                    all_strokes[w][j]['drop']=True
        return all_strokes
    
    def do(self,split_result_dir = None,stroke_data_dir =None,json_save_dir=None,img_save_dir=None,sd_dir=None):
        if split_result_dir == None:
            split_result_dir = self.split_result_dir
        if stroke_data_dir == None:
            stroke_data_dir = self.stroke_data_dir
        if json_save_dir == None:
            json_save_dir = self.json_save_dir
            
        if not os.path.isdir(json_save_dir):
            os.makedirs(json_save_dir)
            
        if img_save_dir:
            if not os.path.isdir(img_save_dir):
                os.makedirs(img_save_dir)

        self.split_result_dir = split_result_dir

        self.all_type = os.listdir(self.split_result_dir)

        self.font_stroke_data = self.load_stroke_dataset(stroke_data_dir)

        self.min_pixel = self.count_min_pixel(self.font_stroke_data)

        for root, dirs, files in os.walk(os.path.join(self.split_result_dir,'0')):
            for progress,file in enumerate(tqdm(files)):
                
                self.progress=progress/len(files)
                refs = self.extract_stroke_refs(file)

                all_strokes=[[] for k in range(self.top_num*2)]

                for ref in refs:
                    target = None
                    candidates = self.find_candidates(ref)

                    rescale_candidates=[]
                    for (_,candidate)in candidates:
                        target = new_rescale(ref,candidate)     

                        ori_img=rasterize_cubic([deepcopy(target['contour'])],self.upm,(self.upm,self.upm))
                        _, target_img = cv2.threshold(ori_img, 128, 255, cv2.THRESH_BINARY)
                        dist = compute_l1_loss(target_img,ref['img'])

                        bi_target_img = deepcopy(target_img)/255
                        rescale_candidates.append({'contour':target['contour'],'bi_img':bi_target_img,'img':ori_img,'dist':dist,'drop':False})
                        
                    if len(rescale_candidates)==0:
                        continue
                    else:
                        for i in range(len(rescale_candidates)):
                            all_strokes[i].append(rescale_candidates[i])


                all_strokes = self.overlap_filter(all_strokes)
                
                temp={i:[] for i in range(len(all_strokes[0]))}
                for all_stroke in all_strokes:
                    for i,stroke in enumerate(all_stroke):
                        if stroke['drop']:
                            continue
                        else:
                            temp[i].append(stroke)
                result = {}
                i=0
                for values in temp.values():
                    if values:
                        result[i]=values
                        i+=1

                if img_save_dir :
                    temps = all_strokes
                    all_strokes=[]
                    for temp in temps:
                        all_stroke=[]
                        for stroke in temp:
                            if stroke['drop']:
                                continue
                            all_stroke.append(stroke['contour'])
                        all_strokes.append(all_stroke)

                    if sd_dir:
                        sd_img = 255-color.rgb2gray(io.imread(os.path.join(sd_dir,file.split('.')[0]+'.png')))*255
                        for i in range(len(all_strokes)):
                            assemble_img = rasterize_cubic(all_strokes[i],self.upm,(self.upm,self.upm))
                            io.imsave(os.path.join(img_save_dir,file.split('.')[0]+'_'+str(i)+'.png'),np.concatenate((sd_img,assemble_img),1))
                    else:
                        for i in range(len(all_strokes)):
                            assemble_img = rasterize_cubic(all_strokes[i],self.upm,(self.upm,self.upm))
                            io.imsave(os.path.join(img_save_dir,file.split('.')[0]+'_'+str(i)+'.png'),assemble_img)

                for key,value in result.items():
                    for i in range(len(value)):
                        del value[i]['img']
                        del value[i]['bi_img']
                        del value[i]['drop']

                with open(os.path.join(json_save_dir,file.split('.')[0]+'.json'), "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
        self.progress = 1
        self.is_finished = True
        return json_save_dir