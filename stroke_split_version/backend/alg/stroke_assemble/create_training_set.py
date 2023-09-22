import json,os
import cv2
import numpy as np
from skimage import io,color
from contour_tools import rasterize_cubic,matrix2contour
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
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
class Create_Training_Set(object):
    def __init__(self):
        self.is_finished = False
        self.progress = 0
        self.upm=512
        self.uinicode = 0
        self.label_colors = {
                0: (0, 0, 0),  # 未标注
                255: (1, 1, 1)
            }
    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress
    
    def set_moments_dir(self,moment_dir):
        self.moment_dir = moment_dir
        
    def set_save_dir(self,save_dir):
        self.save_dir = save_dir

    def get_type(self,root,file):
        path = os.path.join(root, file)
        f = open(path, "r", encoding="utf-8") 
        character = json.load(f)
        if type(character) == dict:
            character = character['info']

        for stroke in character:
            self.all_type.add(merge[stroke['type']])

    def create_data(self,root,file):

        if not os.path.isdir(os.path.join(self.img_save_dir,'all')):
            os.makedirs(os.path.join(self.img_save_dir,'all'))

        path = os.path.join(root, file)
        f = open(path, "r", encoding="utf-8") 
        character = json.load(f)
        if type(character) == dict:
            character = character['info']
        contours = list(map(lambda x: matrix2contour(x['matrix']), character))
        character_img =  color.grey2rgb(rasterize_cubic(contours, self.upm, (self.upm, self.upm)))
        img_path = os.path.join(self.img_save_dir, 'all', file.split('.')[0]+str(self.uinicode)+'.png')
        io.imsave(img_path,character_img)

        label_stroke = {}
        for stroke in character:
            if merge[stroke['type']] not in label_stroke.keys():
                label_stroke[merge[stroke['type']]] = [matrix2contour(stroke['matrix'])]
            else:
                label_stroke[merge[stroke['type']]].append(matrix2contour(stroke['matrix']))

        for stroke_type in list( self.all_type):
            if not os.path.isdir(os.path.join(self.img_save_dir,str(stroke_type))):
                os.makedirs(os.path.join(self.img_save_dir,str(stroke_type)))

            if stroke_type in label_stroke.keys():
                input_img = rasterize_cubic(label_stroke[stroke_type],512,(512,512))
                _, input_img = cv2.threshold(input_img, 128, 255, cv2.THRESH_BINARY)
                label_img = np.zeros((input_img.shape[0], input_img.shape[1], 3), dtype=np.uint8)
                for label, label_color in self.label_colors.items():
                    label_img[input_img == label] = label_color
            else:
                label_img = np.zeros((512, 512, 3), dtype=np.uint8)
            io.imsave(os.path.join(self.img_save_dir,str(stroke_type), file.split('.')[0]+str(self.uinicode)+'.png'),label_img)

        self.uinicode+=1

    def do(self,json_dirs = None,save_dir = None):
        if json_dirs == None:
            json_dirs = self.moment_dir
        if save_dir == None:
            save_dir = self.save_dir
            
        self.img_save_dir = save_dir
        self.all_type = set()
        self.full_progress = 0

        if not os.path.isdir(os.path.join(self.img_save_dir)):
            os.makedirs(os.path.join(self.img_save_dir))
        for data_dir in json_dirs:
            for root, dirs, files in os.walk(data_dir):
                for file in tqdm(files):
                    self.full_progress+=1
                    self.get_type(root,file)

        self.data_train = {}
        self.data_val = {}
        current_progress = 0
        for data_dir in json_dirs:
            for root, dirs, files in os.walk(data_dir):
                for file in tqdm(files):
                    current_progress+=1
                    self.progress = current_progress/len(files)
                    self.create_data(root,file)
        
        self.is_finished = True
        self.progress =1
        return save_dir
