import os
import sys
import json
import numpy as np
from tqdm import tqdm
from contour_tools import rasterize_cubic,matrix2contour

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
    assert union_pixel <= pixel1+pixel2
    assert overlap_pixel <= min(pixel1,pixel2,union_pixel)
    result = {
        'overlap_pixel':overlap_pixel,
        'iou_pixel':overlap_pixel/union_pixel,
        'iou1_pixel':overlap_pixel/pixel1,
        'iou2_pixel':overlap_pixel/pixel2
        }
    return result

def calculate_bbox_ranger(rect):
    x1, x2, y1, y2 = rect.values()
    return (x2-x1),(y2-y1)

def calculate_overlap_area(rect1, rect2):
    x1, x2, y1, y2 = rect1.values()
    x3, x4, y3, y4 = rect2.values()
    # Calculate the horizontal overlap length.
    horizontal_overlap = max(0, min(x2, x4) - max(x1, x3))
    
    # Calculate the vertical overlap length.
    vertical_overlap = max(0, min(y2, y4) - max(y1, y3))

    # Calculate the area of the overlapping part.
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

class Build_Stokre_Dataset(object):
    def __init__(self):
        self.is_finished = False
        self.progress = 0

    def set_moments_dir(self,moment_dir):
        self.moment_dir = moment_dir
        
    def set_save_dir(self,save_dir):
        self.save_dir = save_dir
        
    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress
    def update_stroke_realation_overlap_info(self,stroke_type1,stroke_type2,overlap_area_info,overlap_pixel_info):
        overlap_info ={}
        overlap_info.update(overlap_area_info)
        overlap_info.update(overlap_pixel_info)

        if overlap_info['overlap_pixel'] or overlap_info['overlap_area']:
            # Construct keywords based on two stroke categories, with the category with the smaller number coming first and the larger one coming second.
            if stroke_type1 <= stroke_type2:
                id = str(stroke_type1)+'_'+str(stroke_type2)
            else:
                id = str(stroke_type2)+'_'+str(stroke_type1)
                overlap_info['iou1_area'],overlap_info['iou2_area'] = overlap_info['iou2_area'],overlap_info['iou1_area'] 
                overlap_info['iou1_pixel'],overlap_info['iou2_pixel'] = overlap_info['iou2_pixel'],overlap_info['iou1_pixel'] 

            if id in self.dataset['basic_information']['stroke_relation'].keys():
                for key,value in overlap_info.items():
                    self.dataset['basic_information']['stroke_relation'][id][key].append(value)
            else:
                self.dataset['basic_information']['stroke_relation'][id]={}
                for key,value in overlap_info.items():
                    self.dataset['basic_information']['stroke_relation'][id][key]=[value]
                    
    def update_stroke_realation(self,infos):
        # Count the pixel coverage between different strokes within a character, including bounding box coverage relationships.
        for i in range(len(infos)):
            stroke1 = infos[i]
            stroke_type1 = stroke1["type"]
            bbox1 = stroke1["bbox"]
            for j in range(i+1,len(infos)):
                stroke2 = infos[j]
                stroke_type2 = stroke2["type"]
                bbox2 = stroke2["bbox"]

                # Calculate the pixel coverage between any two strokes, including bounding box coverage relationships.
                overlap_area_info =  calculate_overlap_area(bbox1,bbox2)
                overlap_pixel_info =  calculate_overlap_pixel(stroke1,stroke2)
                self.update_stroke_realation_overlap_info(stroke_type1,stroke_type2,overlap_area_info,overlap_pixel_info)

    
    def update_dataset(self,infos,path):
        stroke_num = len(infos)
        
        # Create a dictionary where the number of strokes serves as the key, and the corresponding characters serve as the values.
        if stroke_num in self.dataset['reference_by_stroke_num'].keys():
            self.dataset['reference_by_stroke_num'][stroke_num].append(path)
        else:
            self.dataset['reference_by_stroke_num'][stroke_num] = [path]
        
        # Count each type of stroke.
        total_types= set()
        for info in infos:
            stroke_type = info["type"]
            total_types.add(stroke_type)
            x_range,y_range = calculate_bbox_ranger(info['bbox'])
            if stroke_type in self.dataset['vector_data'].keys():
                self.dataset['vector_data'][stroke_type]['pixel'].append(info['pixel'])
                self.dataset['vector_data'][stroke_type]['bbox_x_range'].append(x_range)
                self.dataset['vector_data'][stroke_type]['bbox_y_range'].append(y_range)
                self.dataset['vector_data'][stroke_type]['stroke_num'] += 1
                self.dataset['vector_data'][stroke_type]['contour'].append(info)
            else:
                self.dataset['vector_data'][stroke_type] = {
                    'pixel':[info['pixel']],
                    'bbox_x_range':[x_range],
                    'bbox_y_range':[y_range],
                    'stroke_num':1,
                    'stroke_freq':0,
                    'contour':[info]
                    }
                    
        # Count the frequency of strokes appearing in characters, considering only whether a stroke has appeared or not, regardless of the quantity of strokes.
        for stroke_type in list(total_types):
            self.dataset['vector_data'][stroke_type]['stroke_freq']+=1
        
        # Count the pixel coverage between different strokes within characters, including bounding box coverage relationships.
        self.update_stroke_realation(infos)

    def summarize(self):
        # Calculate the frequency distribution of the number of strokes in different Chinese characters.
        for key,value in self.dataset['reference_by_stroke_num'].items():
            self.dataset['basic_information']['stroke_num_distribution'][key] = len(value)

        # Count the coverage relationships between strokes.
        for relation_id in self.dataset['basic_information']['stroke_relation'].keys():
            summary = {}
            summary['num'] = len(self.dataset['basic_information']['stroke_relation'][relation_id]['overlap_area'])
            for key,value in self.dataset['basic_information']['stroke_relation'][relation_id].items():
                summary[key]={
                    'max':max(value),
                    'min':min(value),
                    'average':sum(value)/len(value),
                }
            self.dataset['basic_information']['stroke_relation'][relation_id] = summary
        
        # Collect stroke data for each category, including minimum value and bounding box dimensions
        for stroke_type in self.dataset['vector_data'].keys():

            self.dataset['vector_data'][stroke_type]['pixel'] = {
                'max':max(self.dataset['vector_data'][stroke_type]['pixel']),
                'min':min(self.dataset['vector_data'][stroke_type]['pixel']),
                'average':sum(self.dataset['vector_data'][stroke_type]['pixel'])/len(self.dataset['vector_data'][stroke_type]['pixel']),
                }

            self.dataset['vector_data'][stroke_type]['bbox_x_range'] = {
                'max':max(self.dataset['vector_data'][stroke_type]['bbox_x_range']),
                'min':min(self.dataset['vector_data'][stroke_type]['bbox_x_range']),
                'average':sum(self.dataset['vector_data'][stroke_type]['bbox_x_range'])/len(self.dataset['vector_data'][stroke_type]['bbox_x_range']),
                }
            self.dataset['vector_data'][stroke_type]['bbox_y_range'] = {
                'max':max(self.dataset['vector_data'][stroke_type]['bbox_y_range']),
                'min':min(self.dataset['vector_data'][stroke_type]['bbox_y_range']),
                'average':sum(self.dataset['vector_data'][stroke_type]['bbox_y_range'])/len(self.dataset['vector_data'][stroke_type]['bbox_y_range']),
                }
            
    def do(self,json_dir = None,save_dir = None,is_merged=True):
        if json_dir == None:
            json_dir = self.moment_dir
        if save_dir == None:
            save_dir = self.save_dir
        if not os.path.isdir(os.path.join(save_dir)):
            os.makedirs(os.path.join(save_dir))
        self.dataset = {}
        self.dataset['basic_information']={'stroke_num_distribution':{},'stroke_relation':{}}
        self.dataset['reference_by_stroke_num']={}
        self.dataset['vector_data']={}

        for root, dirs, files in os.walk(json_dir):
            for progress,file in enumerate(tqdm(files)):
                self.progress = (progress+1)/len(files)

                if file[0] == ".":
                    continue

                path = os.path.join(root, file)
                f = open(path, "r", encoding="utf-8") 
                data = json.load(f)
                
                infos = data['info']
                
                self.update_dataset(infos,path)
        
        self.summarize()
        with open(os.path.join(save_dir, 'stroke_dataset.json'), "w", encoding="utf-8") as f:
            json.dump(self.dataset,f, indent=4)

        self.is_finished = True
        self.progress=1
        return save_dir