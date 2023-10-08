# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
from torch import nn
from torchvision import transforms

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img
def draw_single_char_new(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
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
    # img = nn.ZeroPad2d(m)(img) 
    img = img.squeeze(0)  
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img

def draw_example(ch, src_font, dst_imgpath, canvas_size, x_offset, y_offset, filter_hashes):
    #dst_img = draw_single_char_new(ch, dst_font, canvas_size, x_offset, y_offset)
    dst_img = Image.open(dst_imgpath).convert("RGB")
    dst_img = dst_img.resize((canvas_size,canvas_size),Image.ANTIALIAS)
    # check the filter example in the hashes or not
    src_img = draw_single_char_new(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    example_img = example_img.convert('L')
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char_new(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2img(dst_imgfolder, sample_dir, src="./data/font_src",  char_size =150, canvas_size=256,
             x_offset=20, y_offset=20, sample_count=100,  label=6, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    # To delete the files inside a folder to prevent contamination
    for root,dirs,files in os.walk(sample_dir,topdown=False):
        for name in files:
            os.remove(os.path.join(root,name))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    filter_hashes = set()
    if filter_by_hash:
        pass
        #filter_hashes = set(filter_recurring_hash(charset, src_font, canvas_size, x_offset, y_offset))
        #print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0
    
    clist = []
    for item in os.listdir(dst_imgfolder):
        '''
        if count == sample_count:
            break
        '''
        dst_imgpath = os.path.join(dst_imgfolder,item)
        c = item.split('.')[0]
        clist.append(c)
        print(dst_imgpath)
        e = draw_example(c, src_font, dst_imgpath, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)
    return clist

def draw_example_new(ch, src_font, canvas_size, x_offset, y_offset, filter_hashes):
    src_img = draw_single_char_new(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (canvas_size, 0))
    example_img = example_img.convert('L')
    return example_img

def font2imgtest(charset,sample_dir,sample_count,src ="./data/font_src/SimSun.ttf",  char_size =150, canvas_size=256,
             x_offset=20, y_offset=20,   label=6, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, src_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_example_new(c, src_font, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)
