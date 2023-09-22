import numpy as np
import random,json,os
from skimage import io
from copy import deepcopy
from contour_tools import rasterize_cubic,matrix2contour

from PIL import Image
from torchvision import transforms
from torch import nn

params = {
    '0': {'name': 'H', 'x_factor': [0.5, 2], 'y_factor': [0.9, 1.1], 'rotation': [-0.1, 0.1]}, 
    '1': {'name': 'S', 'x_factor': [0.9, 1.1], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '2': {'name': 'P', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '3': {'name': 'N', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '4': {'name': 'D', 'x_factor': [0.8, 1.2], 'y_factor': [0.8, 1.2], 'rotation': [-5, 5]}, 
    '5': {'name': 'T', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '6': {'name': 'HZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '7': {'name': 'HP', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '8': {'name': 'HG', 'x_factor': [0.5, 2], 'y_factor': [0.8, 1.2], 'rotation': [-5, 5]}, 
    '9': {'name': 'SZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '10': {'name': 'ST', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '11': {'name': 'SW', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '12': {'name': 'SG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '13': {'name': 'WG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '14': {'name': 'XG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '15': {'name': 'PZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '16': {'name': 'BXG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '17': {'name': 'PD', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '18': {'name': 'HZG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '19': {'name': 'SWG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '20': {'name': 'HZWG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '21': {'name': 'SZZG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '22': {'name': 'HPWG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '23': {'name': 'HZT', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '24': {'name': 'HZW', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '25': {'name': 'HZZZG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '26': {'name': 'HXG', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '27': {'name': 'HZZP', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '28': {'name': 'SZP', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '29': {'name': 'SZZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '30': {'name': 'HZZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '31': {'name': 'HZZZ', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '32': {'name': 'ZD', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '33': {'name': 'PP', 'x_factor': [0.5, 2], 'y_factor': [0.8, 1.2], 'rotation': [-5, 5]}, 
    '34': {'name': 'S', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '35': {'name': 'SP1', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '36': {'name': 'SP2', 'x_factor': [0.5, 2], 'y_factor': [0.5, 2], 'rotation': [-5, 5]}, 
    '37': {'name': 'NA', 'x_factor': [0.9, 1.1], 'y_factor': [0.9, 1.1], 'rotation': [-0.1, 0.1]}
    }


def calculate_overlap_area(rect1, rect2):
    x1, x2, y1, y2 = rect1.values()
    x3, x4, y3, y4 = rect2.values()
    # 计算水平方向上的重叠长度
    horizontal_overlap = max(0, min(x2, x4) - max(x1, x3))
    
    # 计算垂直方向上的重叠长度
    vertical_overlap = max(0, min(y2, y4) - max(y1, y3))

    # 计算重叠部分的面积
    overlap_area = horizontal_overlap * vertical_overlap
    area1 = (x2-x1)*(y2-y1)
    area2 = (x4-x3)*(y4-y3)
    union_area = area1+area2-overlap_area
    assert union_area <= area1+area2
    assert overlap_area <= min(area1,area2,union_area)
    result = {
        'overlap_area':overlap_area,
        'iou_area':overlap_area/union_area,
        'iou1_area':overlap_area/area1,
        'iou2_area':overlap_area/area2
        }
    return result

def calculate_overlap_pixel(stroke1,stroke2,upm=512):
    contour1 = [matrix2contour(stroke1['matrix'])]
    contour2 = [matrix2contour(stroke2['matrix'])]
    img1 =  rasterize_cubic(contour1, upm, (upm, upm))
    img2 =  rasterize_cubic(contour2, upm, (upm, upm))
    img_union = rasterize_cubic(contour1+contour2, upm, (upm, upm))

    pixel1 = np.sum(img1)/255
    pixel2 = np.sum(img2)/255
    union_pixel = min(np.sum(img_union)/255,pixel1+pixel2)
    overlap_pixel = max(0,pixel1+pixel2-union_pixel)
    if overlap_pixel<2:
        overlap_pixel=0
    #print(overlap_pixel,pixel1,pixel2,union_pixel)
    assert union_pixel <= pixel1+pixel2
    #assert overlap_pixel <= min(pixel1,pixel2,union_pixel)+1
    result = {
        'overlap_pixel':overlap_pixel,
        'iou_pixel':overlap_pixel/union_pixel,
        'iou1_pixel':overlap_pixel/pixel1,
        'iou2_pixel':overlap_pixel/pixel2
        }
    return result

def cal_bbox_size(stroke,coordinate,x_bias,y_bias,upm=512):
    transfer = np.array([[1,0,0],
                            [0,1,0],
                            [x_bias,y_bias,1]])
    
    coordinate = coordinate@transfer
    
    stroke['matrix']['cordinate'] = coordinate.tolist()
    img = rasterize_cubic([matrix2contour(stroke['matrix'])], upm, (upm, upm))
    img = Image.fromarray(img)
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    return ({'x_min':l-x_bias,'x_max':r-x_bias,'y_min':u-y_bias,'y_max':d-y_bias})

class Data_Augmentation(object):
    def __init__(self,data_threshold=150,stroke_num_threshold=[10,15],max_try=50):
        self.data_threshold = data_threshold
        self.stroke_num_threshold = stroke_num_threshold
        self.max_try = max_try
        self.is_finished = False
        self.progress = 0
        self.upm = 512

    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress

    def get_dataset_info(self,dataset_path):
        stroke_dataset = open(os.path.join(dataset_path,'stroke_dataset.json'), "r", encoding="utf-8") 
        self.stroke_dataset=json.load(stroke_dataset)

        stroke_num = {}
        stroke_freq = {}
        for stroke_type in self.stroke_dataset['vector_data'].keys():
            stroke_num[stroke_type] = self.stroke_dataset['vector_data'][stroke_type]['stroke_num']
            stroke_freq[stroke_type] = self.stroke_dataset['vector_data'][stroke_type]['stroke_freq']


        self.weight_total = stroke_num
        self.stroke_freq_rare = {}
        for stroke_type in stroke_freq.keys():
            if stroke_freq[stroke_type] < self.data_threshold:
                self.stroke_freq_rare[stroke_type] = stroke_freq[stroke_type]

        self.weight_rare = {}
        for stroke_type in self.stroke_freq_rare.keys():
            self.weight_rare[stroke_type] = sum(list(self.stroke_freq_rare.values()))/self.stroke_freq_rare[stroke_type]

    def update_weight(self,add_type):
        for stroke_type in list(add_type):
            self.stroke_freq_rare[stroke_type] += 1

        for stroke_type in self.stroke_freq_rare.keys():
            self.weight_rare[stroke_type] = sum(list(self.stroke_freq_rare.values()))/self.stroke_freq_rare[stroke_type]

    def update_weight_new(self,character):
        add_type =set()
        for stroke in character:
            add_type.add(stroke['type'])
        for stroke_type in list(add_type):
            if str(stroke_type) in self.stroke_freq_rare.keys():
                self.stroke_freq_rare[str(stroke_type)] += 1

        for stroke_type in self.stroke_freq_rare.keys():
            self.weight_rare[stroke_type] = sum(list(self.stroke_freq_rare.values()))/self.stroke_freq_rare[stroke_type]


    def stroke_validation(self,character,stroke):
        
        for current_stroke in character:
            if current_stroke == []:
                continue
            overlap_area_info = calculate_overlap_area(current_stroke['bbox'],stroke['bbox'])
            overlap_pixel_info =  calculate_overlap_pixel(stroke,current_stroke)

            overlap_info ={}
            overlap_info.update(overlap_area_info)
            overlap_info.update(overlap_pixel_info)

            if stroke['type'] <= current_stroke['type']:
                id = str(stroke['type'])+'_'+str(current_stroke['type'])
            else:
                id = str(current_stroke['type'])+'_'+str(stroke['type'])
                overlap_info['iou1_area'],overlap_info['iou2_area'] = overlap_info['iou2_area'],overlap_info['iou1_area'] 
                overlap_info['iou1_pixel'],overlap_info['iou2_pixel'] = overlap_info['iou2_pixel'],overlap_info['iou1_pixel'] 
                
            if overlap_info['iou_area']>0 or overlap_info['iou_pixel']>0:
                if id not in self.stroke_dataset['basic_information']['stroke_relation'].keys():
                    return False
                elif overlap_info['iou_area'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou_area']['max']:
                    return False
                elif overlap_info['iou_pixel'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou_pixel']['max']:
                    return False
                elif overlap_info['iou1_area'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou1_area']['max']:
                    return False
                elif overlap_info['iou2_area'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou2_area']['max']:
                    return False
                elif overlap_info['iou1_pixel'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou1_pixel']['max']:
                    return False
                elif overlap_info['iou2_pixel'] > self.stroke_dataset['basic_information']['stroke_relation'][id]['iou2_pixel']['max']:
                    return False
                
        return True
            
    def generate_stroke(self,stroke_type):
        stroke_ori = random.choice(self.stroke_dataset['vector_data'][stroke_type]['contour'])
        stroke_new = deepcopy(stroke_ori)

        # 1,根据包围盒将矢量移动到关于原点对称
        x_bias_ori = (stroke_ori['bbox']['x_max']+stroke_ori['bbox']['x_min'])/2
        y_bias_ori = (stroke_ori['bbox']['y_max']+stroke_ori['bbox']['y_min'])/2

        transfer_1 = np.array([[1,0,0],
                              [0,1,0],
                              [-x_bias_ori,-y_bias_ori,1]])
        
        coordinate_central = stroke_ori['matrix']['cordinate']@transfer_1

        # 2，根据进行随即缩放和旋转
        x_range = stroke_ori['bbox']['x_max']-stroke_ori['bbox']['x_min']
        y_range = stroke_ori['bbox']['y_max']-stroke_ori['bbox']['y_min']

        x_factor_max = min(0.95*self.upm/x_range, params[stroke_type]['x_factor'][1])
        y_factor_max = min(0.95*self.upm/y_range, params[stroke_type]['y_factor'][1])

        x_factor_min = max(0.8*self.stroke_dataset['vector_data'][stroke_type]['bbox_x_range']['min']/x_range, params[stroke_type]['x_factor'][0])
        y_factor_min = max(0.8*self.stroke_dataset['vector_data'][stroke_type]['bbox_y_range']['min']/y_range, params[stroke_type]['y_factor'][0])

        # print(self.stroke_dataset['vector_data'][stroke_type]['bbox_x_range']['min'],x_range,stroke_type)
        # print(self.stroke_dataset['vector_data'][stroke_type]['bbox_y_range']['min'],y_range,stroke_type)
        # try:
        #     assert max([x_factor_min,y_factor_min]) <= 1
        # except AssertionError:
        #     img = rasterize_cubic([matrix2contour(stroke_ori['matrix'])], self.upm, (self.upm, self.upm))
        #     io.imsave('test.png',img)
        #     input()
        
        x_factor = random.uniform(x_factor_min, x_factor_max)
        y_factor = random.uniform(y_factor_min, y_factor_max)

        theta = np.radians(random.uniform(params[stroke_type]['rotation'][0], params[stroke_type]['rotation'][1]))

        transfer_2 = np.array([[x_factor*np.cos(theta),x_factor*np.sin(theta),0],
                              [-y_factor*np.sin(theta),y_factor*np.cos(theta),0],
                              [0,0,1]])
        coordinate_resize = coordinate_central@transfer_2

        stroke_new['bbox'] = cal_bbox_size(deepcopy(stroke_new),deepcopy(coordinate_resize),self.upm//2,self.upm//2,self.upm)

        x_bias_new = random.uniform(0-stroke_new['bbox']['x_min'],self.upm-stroke_new['bbox']['x_max'])
        y_bias_new = random.uniform(0-stroke_new['bbox']['y_min'],self.upm-stroke_new['bbox']['y_max'])

        transfer_3 = np.array([[1,0,0],
                              [0,1,0],
                              [x_bias_new,y_bias_new,1]])
        
        coordinate_result = coordinate_resize@transfer_3
        
        stroke_new['matrix']['cordinate'] = coordinate_result.tolist()
        
        stroke_new['bbox']['x_max'] += x_bias_new
        stroke_new['bbox']['x_min'] += x_bias_new
        stroke_new['bbox']['y_max'] += y_bias_new
        stroke_new['bbox']['y_min'] += y_bias_new

        return stroke_new
        
    def replace_stroke(self,stroke_old,new_stroke_type):
        stroke_slect = deepcopy(random.choice(self.stroke_dataset['vector_data'][new_stroke_type]['contour']))

        x_bias_old = (stroke_old['bbox']['x_max']+stroke_old['bbox']['x_min'])/2
        y_bias_old = (stroke_old['bbox']['y_max']+stroke_old['bbox']['y_min'])/2
        x_bias_slect = (stroke_slect['bbox']['x_max']+stroke_slect['bbox']['x_min'])/2
        y_bias_slect = (stroke_slect['bbox']['y_max']+stroke_slect['bbox']['y_min'])/2

        x_range_old = stroke_old['bbox']['x_max']-stroke_old['bbox']['x_min']
        y_range_old = stroke_old['bbox']['y_max']-stroke_old['bbox']['y_min']
        x_range_slect = stroke_slect['bbox']['x_max']-stroke_slect['bbox']['x_min']
        y_range_slect = stroke_slect['bbox']['y_max']-stroke_slect['bbox']['y_min']

        x_factor_max = params[new_stroke_type]['x_factor'][1]
        y_factor_max = params[new_stroke_type]['y_factor'][1]

        x_factor_min = max(0.8*self.stroke_dataset['vector_data'][new_stroke_type]['bbox_x_range']['min']/x_range_slect, params[new_stroke_type]['x_factor'][0])
        y_factor_min = max(0.8*self.stroke_dataset['vector_data'][new_stroke_type]['bbox_y_range']['min']/y_range_slect, params[new_stroke_type]['y_factor'][0])

        x_factor = max(min(x_factor_max, x_range_old/x_range_slect), x_factor_min)
        y_factor = max(min(y_factor_max, y_range_old/y_range_slect), y_factor_min)

        transfer_1 = np.array([[1,0,0],
                              [0,1,0],
                              [-x_bias_slect,-y_bias_slect,1]])
        
        coordinate_central = stroke_slect['matrix']['cordinate']@transfer_1

        transfer_2 = np.array([[x_factor,0,0],
                              [0,y_factor,0],
                              [0,0,1]])
        coordinate_resize = coordinate_central@transfer_2

        stroke_slect['bbox'] = cal_bbox_size(deepcopy(stroke_slect),deepcopy(coordinate_resize),self.upm//2,self.upm//2,self.upm)

        x_bias_new = max(min(self.upm-stroke_slect['bbox']['x_max'], x_bias_old), 0-stroke_slect['bbox']['x_min'])
        y_bias_new = max(min(self.upm-stroke_slect['bbox']['y_max'], y_bias_old), 0-stroke_slect['bbox']['y_min'])

        transfer_3 = np.array([[1,0,0],
                              [0,1,0],
                              [x_bias_new,y_bias_new,1]])
        
        coordinate_result = coordinate_resize@transfer_3
        
        stroke_slect['matrix']['cordinate'] = coordinate_result.tolist()

        stroke_slect['bbox']['x_max'] += x_bias_new
        stroke_slect['bbox']['x_min'] += x_bias_new
        stroke_slect['bbox']['y_max'] += y_bias_new
        stroke_slect['bbox']['y_min'] += y_bias_new

        return stroke_slect

    def random_layout(self,stroke_num):
        rare_num = stroke_num//2

        # add_type = set()

        character = []
        for i in range(rare_num):
            j=0
            while(True):
                stroke_type = random.choices(list(self.weight_rare.keys()),weights=list(self.weight_rare.values()))[0]
                stroke =self.generate_stroke(stroke_type)

                if self.stroke_validation(character,stroke):
                    character.append(stroke)
                    # add_type.add(stroke_type)
                    break
                j+=1
                if j == self.max_try:
                    return character

        for i in range(stroke_num-rare_num):
            j=0
            while(True):
                stroke_type = random.choices(list(self.weight_total.keys()),weights=list(self.weight_total.values()))[0]
                stroke = self.generate_stroke(stroke_type)

                if self.stroke_validation(character,stroke):
                    character.append(stroke)
                    break
                j+=1
                if j == self.max_try:
                    return character

        # self.update_weight(add_type)
        
        return character
    
    def random_replace(self,stroke_num):
        character_path = random.choices(list(self.stroke_dataset['reference_by_stroke_num'][str(stroke_num)]))[0]
        f = open(character_path, "r", encoding="utf-8") 
        character = json.load(f)['info']

        # add_type = set()

        replace_num = random.randint(1, max(1,stroke_num//4))
        replace_indexs = []
        for i in range(replace_num):
            while(True):
                index = random.randint(0,len(character)-1)
                if index not in replace_indexs:
                    replace_indexs.append(index)
                    break
            stroke_old = character[index]

            j=0
            while(True):
                new_stroke_type = random.choices(list(self.weight_rare.keys()),weights=list(self.weight_rare.values()))[0]
                stroke_new = self.replace_stroke(stroke_old,new_stroke_type)
                character[index] = []

                if self.stroke_validation(character,stroke_new):
                    character[index] = stroke_new
                    # add_type.add(new_stroke_type)
                    break
                j+=1
                if j == self.max_try:
                    character[index] = stroke_old
                    if i == 0:
                        return None
                    else:
                        return character
                
        # self.update_weight(add_type)

        return character
                                  
    def random_add(self,stroke_num):
        character_path = random.choices(self.stroke_dataset['reference_by_stroke_num'][str(stroke_num)])[0]
        f = open(character_path, "r", encoding="utf-8") 
        character = json.load(f)['info']

        # add_type = set()

        add_num = random.randint(1, 3)
        for i in range(add_num):
            j=0
            while(True):
                stroke_type = random.choices(list(self.weight_rare.keys()),weights=list(self.weight_rare.values()))[0]
                stroke = self.generate_stroke(stroke_type)
                if self.stroke_validation(character,stroke):
                    character.append(stroke)
                    # add_type.add(stroke_type)
                    break
                j+=1
                if j == self.max_try:
                    if i == 0:
                        return None
                    else:
                        return character
                
        # self.update_weight(add_type)
        
        return character

    def generate_character(self):
        stroke_num = int(random.choices(list(self.stroke_dataset['basic_information']['stroke_num_distribution'].keys()),
                                            weights=list(self.stroke_dataset['basic_information']['stroke_num_distribution'].values()))[0])
        
        if stroke_num <= self.stroke_num_threshold[0]:
            augment_type = random.randint(1,4)
            if augment_type==1 or augment_type==2:
                character = self.random_layout(stroke_num)
            elif augment_type==3:
                character = self.random_replace(stroke_num)
            elif augment_type==4:
                character = self.random_add(stroke_num)
        elif stroke_num <= self.stroke_num_threshold[1]:
            augment_type = random.randint(1,3)
            if augment_type==1:
                character = self.random_layout(stroke_num)
            elif augment_type==2:
                character = self.random_replace(stroke_num)
            elif augment_type==3:
                character = self.random_add(stroke_num)
        else:
            augment_type = random.randint(2,3)
            if augment_type==2:
                character = self.random_replace(stroke_num)
            elif augment_type==3:
                character = self.random_add(stroke_num)
        
        return character

    def set_dataset_path(self,dataset_path):
        self.dataset_path = dataset_path
    
    def set_save_dir(self,save_dir):
        self.save_dir = save_dir
    
    def set_save_img_dir(self,save_img_dir):
        self.save_img_dir = save_img_dir
        
    def do(self,dataset_path=None,save_dir=None,save_img_dir=None):
        if dataset_path == None:
            dataset_path = self.dataset_path
        if save_dir == None:
            save_dir = self.save_dir
        if save_img_dir == None:
            save_img_dir = self.save_img_dir

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if save_img_dir and (not os.path.isdir(save_img_dir)):
            os.makedirs(save_img_dir)

        self.get_dataset_info(dataset_path)

        i = 0
        while(min(self.stroke_freq_rare.values())<self.data_threshold):
            self.progress=min(min(self.stroke_freq_rare.values())/self.data_threshold,1)
            character = self.generate_character()
            if character == None:
                continue
            self.update_weight_new(character)
            if save_img_dir:
                contours = list(map(lambda x: matrix2contour(x['matrix']), character))
                img = rasterize_cubic(contours, self.upm, (self.upm, self.upm))
                io.imsave(os.path.join(save_img_dir,'%04d'%i+'.png'),img)

            with open(os.path.join(save_dir,'%04d'%i+'.json'), "w", encoding="utf-8") as f:
                json.dump(character, f, indent=4)
            i+=1
            if i%100==0:
                print(min(self.stroke_freq_rare.values()))

        self.progress = 1
        self.is_finished = True
        return save_dir

def main():
    agent = Data_Augmentation(140, stroke_num_threshold=[10,15], max_try=40)

    agent.do('stroke_dataset_exper_3.0/Song.json','agumentation_exper/stroke/Song')
        
if __name__ == '__main__':
    main()