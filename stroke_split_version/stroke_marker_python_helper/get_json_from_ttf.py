from fontTools.ttLib import TTFont
import json,os
from contour_tools import quadratic2cubic_contours
from skimage import io
from tqdm import tqdm
# load font file
def get_char_dict(glyf_table):
    dict ={}
    for key in glyf_table.glyphs.keys():
        if key.startswith('uni') and key != 'union':
            try: 
                dict[chr(int(key[3:], 16))]=key
            except:
                continue
    return dict

def get_contour(glyf_table,char_dict,char):
    glypf = glyf_table[char_dict[char]].getCoordinates(glyf_table)

    contours = []
    contour = []
    for i,(coord,flag) in enumerate(zip(glypf[0],glypf[2])):
        contour.append({'x':coord[0],'y':coord[1],'on':bool(flag)})
        if i in glypf[1]:
            contours.append(contour)
            contour=[]
    return contours

class Json_from_ttf(object):
    def __init__(self):
        self.is_finished = False
        self.progress = 0
        self.char_list =[]
        for line in open('char.txt', "r", encoding='UTF-8'):
            self.char_list.append(line.replace('\n', ''))

    def isfinished(self):
        return self.is_finished
    
    def getprogress(self):
        return self.progress
    
    def do(self,font_path):
        font_name = font_path.split('.')[0]

        if font_path.split('.')[1].upper()!="TTF":
            raise ValueError("Invalid font file type")
        font = TTFont(font_path)
        upm = font['head'].unitsPerEm
        baseline = upm - font['OS/2'].sTypoAscender

        miss_char = []

        data = {}
        data['upm'] = upm
        data['baseline'] = baseline
        dict = font['glyf']

        char_dict = get_char_dict(dict)
        for progress,char in enumerate(tqdm(self.char_list)):
            self.progress = (progress+1)/len(self.char_list)
            if char not in char_dict.keys():
                miss_char.append(char)
                continue
            
            contours = get_contour(dict,char_dict,char)
            data[char]={'contours': quadratic2cubic_contours(contours)}

        f = open(font_name+'.json', "w", encoding="utf-8")
        json.dump(data, f, indent=4)
        self.is_finished=True
        return (font_name+'.json'),miss_char

def main():
    agent = Json_from_ttf()
    agent.do('FZVariable-FengRSTJ.TTF')

if __name__ == '__main__':
    main()