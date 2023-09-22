import copy
import cairo
import numpy as np


def contour2matrix(contour):
    matrix = {}
    ones = [1 for p in contour]
    on = [p['on'] for p in contour]

    x = [p['x'] for p in contour]
    y = [p['y'] for p in contour]
    cordinate = np.array([x, y, ones]).T

    matrix['cordinate'] = cordinate.tolist()
    matrix['on'] = on

    if 'delta_x_min' in contour[0].keys():
        delta_x_min = [p['delta_x_min'] for p in contour]
        delta_y_min = [p['delta_y_min'] for p in contour]
        delta_min = np.array([delta_x_min, delta_y_min, ones]).T
        matrix['delta_min'] = delta_min.tolist()

    if 'delta_x_max' in contour[0].keys():
        delta_x_max = [p['delta_x_max'] for p in contour]
        delta_y_max = [p['delta_y_max'] for p in contour]
        delta_max = np.array([delta_x_max, delta_y_max, ones]).T
        matrix['delta_max'] = delta_max.tolist()
    
    return matrix

def matrix2contour(matrix):
    contour = [{'x':cordinate[0],'y':cordinate[1],
                'on':on} for (cordinate,on) in 
                zip(matrix['cordinate'],matrix['on'])]    
    if 'delta_min' in matrix.keys():
        contour = [{k:v for d in [p,{'delta_x_min':delta_min[0],'delta_y_min':delta_min[1]}] for k,v in d.items()}
                   for (p,delta_min) in zip (contour,matrix['delta_min'])]
    if 'delta_max' in matrix.keys():
        contour = [{k:v for d in [p,{'delta_x_max':delta_min[0],'delta_y_max':delta_min[1]}] for k,v in d.items()}
                   for (p,delta_min) in zip (contour,matrix['delta_max'])]        
    
    return contour


#将矢量点根据包围盒标准化
def resize_contour(contours,bbox,canvas_size):
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)

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

# 处理可变字体
# 对于笔画粗细可变的字体，取默认、最粗、平均值三种构建对应contour
def var_font_filter(var_contours,return_average=False):
    var_flag = False
    for c in var_contours:
        for pt in c:
            if isinstance(pt['x'],int)==False or isinstance(pt['y'],int)==False:
                var_flag = True

    if not var_flag:
        return [var_contours]
    elif return_average:
        contours3 = []
        for c in var_contours:
            contour3 = []

            for pt in c:
                if isinstance(pt['x'],int):
                    x3 = pt['x']
                else:
                    x3 = pt['x'][0] + pt['x'][1]['delta']*0.5

                if isinstance(pt['y'],int):
                    y3 = pt['y']
                else:
                    y3 = pt['y'][0] + pt['y'][1]['delta']*0.5

                contour3.append({ 'x': x3, 'y': y3, 'on': pt['on'] })
            contours3.append(contour3)     
            
        return [contours3]
    else:
        contours1 = []
        contours2 = []
        contours3 = []
        for c in var_contours:
            contour1 = []
            contour2 = []
            contour3 = []

            for pt in c:
                if isinstance(pt['x'],int):
                    x1=x2=x3 = pt['x']
                else:
                    x1 = pt['x'][0]
                    x2 = pt['x'][0] + pt['x'][-1]['delta']
                    x3 = pt['x'][0] + sum([delta['delta'] for delta in pt['x'][1:]])/max(len(pt['x'])-1,2)

                if isinstance(pt['y'],int):
                    y1=y2=y3 = pt['y']
                else:
                    y1 = pt['y'][0]
                    y2 = pt['y'][0] + pt['y'][-1]['delta']
                    y3 = pt['y'][0] + sum([delta['delta'] for delta in pt['y'][1:]])/max(len(pt['y'])-1,2)

                contour1.append({ 'x': x1, 'y': y1, 'on': pt['on'] })
                contour2.append({ 'x': x2, 'y': y2, 'on': pt['on'] })
                contour3.append({ 'x': x3, 'y': y3, 'on': pt['on'] })

            contours1.append(contour1)
            contours2.append(contour2)
            contours3.append(contour3)

    return[contours1,contours2,contours3]

# 将二次型贝塞尔曲线转为三次型
def quadratic2cubic_contours(contours):
    contours = [regularize_quadratic_contour(c) for c in contours]
    for c in contours:
        i = 0
        while i < len(c) - 1:
            if not c[i]['on']:
                raise Exception('Invalid regularized quadratic.')
            if not c[i+1]['on']:
                end_pt = c[i+2] if i+2 < len(c) else c[0]
                cps = quadratic2cubic(
                    c[i]['x'], c[i]['y'], c[i+1]['x'], c[i+1]['y'], end_pt['x'], end_pt['y'])
                del c[i+1]
                p1 = {'x': int(cps[0]), 'y': int(cps[1]), 'on': False}
                p2 = {'x': int(cps[2]), 'y': int(cps[3]), 'on': False}
                c.insert(i+1, p1)
                c.insert(i+2, p2)
                i += 3
            else:
                end_pt = c[i+1] if i+1 < len(c) else c[0]
                i += 1

    for c in contours:
        i = 0
        while i < len(c):
            c[i]['x'] = int(c[i]['x'])
            c[i]['y'] = int(c[i]['y'])
            i += 1
    
    return contours

# 计算二次型贝塞尔曲线转为三次型时，对应的三次型曲线中的两个离轨控制点
def quadratic2cubic(x0, y0, x1, y1, x2, y2):
    cx1 = x0 + (x1-x0)*2/3; cy1 = y0 + (y1-y0)*2/3
    cx2 = x2 + (x1-x2)*2/3; cy2 = y2 + (y1-y2)*2/3
    return ( cx1, cy1, cx2, cy2 )

# 对二次型贝塞尔曲线中省略的在轨控制点
def regularize_quadratic_contour(contour):
    result = []
    if not contour[0]['on']:
        if contour[-1]['on']:
            result.append({'x': float(contour[-1]['x']),
                           'y': float(contour[-1]['y']), 'on': True})
        else:
            first_x = contour[0]['x']
            first_y = contour[0]['y']
            last_x = contour[-1]['x']
            last_y = contour[-1]['y']
            result.append({'x': (first_x + last_x)/2,
                           'y': (first_y + last_y)/2, 'on': True})
    for i, pt in enumerate(contour[:-1]):
        result.append(
            {'x': float(pt['x']), 'y': float(pt['y']), 'on': pt['on']})
        next_pt = contour[i+1]
        if not pt['on'] and not next_pt['on']:
            result.append({'x': (pt['x'] + next_pt['x'])/2,
                           'y': (pt['y'] + next_pt['y'])/2, 'on': True})
    # Do nothing when the last point is moved to the head
    if contour[-1]['on'] and not contour[0]['on']:
        pass
    else:
        result.append({'x': float(contour[-1]['x']),
                       'y': float(contour[-1]['y']), 'on': contour[-1]['on']})

    # Making the top-rightest on-point as the starting point
    max_on_i = 0
    max_x = -float('inf')
    max_y = -float('inf')
    for i, pt in enumerate(result):
        if pt['on']:
            if pt['y'] > max_y:
                max_on_i = i
                max_x = pt['x']
                max_y = pt['y']
            if pt['y'] == max_y and pt['x'] > max_x:
                max_on_i = i
                max_x = pt['x']

    return [] if not result else (result[max_on_i:] + result[:max_on_i])

# 利用向量叉积计算轨迹走向（svg中由于轮廓坐标上下翻转，因此外围曲线是顺时针，这与原始轮廓中正好相反）
def judge_contour_orientation(contour):
    on_pts_x = []
    on_pts_y = []
    for pt in contour:
        on_pts_x.append(pt['x'])
        on_pts_y.append(pt['y'])
    if len(pt) <=2:
        return 'clockwise'
    i = on_pts_x.index(max(on_pts_x))
    
    if i == 0:
        j = -1
    else:
        j = i-1
    
    if i == len(on_pts_x)-1:
        k = 0
    else:
        k = 1+i

    a = np.array([on_pts_x[i] - on_pts_x[j],on_pts_y[i]-on_pts_y[j]])
    b = np.array([on_pts_x[k] - on_pts_x[i],on_pts_y[k]-on_pts_y[i]])

    c = np.cross(a,b)

    if c < 0:
        return 'clockwise'
    elif c > 0:
        return 'counterclockwise'
    elif c == 0 and on_pts_y[j]>on_pts_y[i]:
        return 'clockwise'
    else:
        return 'counterclockwise'

# 根据contour中点的坐标粗略计算包围盒大小，可以通过所有控制点计算也可以仅通过在轨控制点计算
def get_box(contour,mode='simple_any'):
    pts_x = []
    pts_y = []
    
    if mode == 'simple_any':
        for pt in contour:
            pts_x.append(pt['x'])
            pts_y.append(pt['y'])
    elif mode == 'simple_on':
        for pt in contour:
            if pt['on']:
                pts_x.append(pt['x'])
                pts_y.append(pt['y'])  

    x_min = min(pts_x)
    x_max = max(pts_x)
    y_min = min(pts_y)
    y_max = max(pts_y)

    return {'coordinate':(x_min,x_max,y_min,y_max),'area':(x_max-x_min)*(y_max-y_min)}

# 根据包围盒坐标计算一个contour是否在另一个contour内部
def box1_in_box2(box1,box2):
    x1_min, x1_max, y1_min, y1_max = box1['coordinate']
    x2_min, x2_max, y2_min, y2_max = box2['coordinate']
    
    if x1_min >= x2_min and x1_max <= x2_max and y1_min >= y2_min and y1_max <= y2_max:
        return True
    else:
        return False

# 建立外轮廓与内轮廓的连接关系，同一连通域内，内轮廓的uplinked是外轮廓，没有downlink，外轮廓的uplink是自己本身，downlink是自己与所有内轮廓
def get_contour_link(contours):
    contour_orientation=[]
    contour_link = {}
    for i,c in enumerate(contours):
        contour_link[i]={'uplink':None,'downlink':None}
        orientation = judge_contour_orientation(c)
        if orientation=='clockwise':
            contour_link[i]['uplink']=i
            contour_link[i]['downlink']=[i]
            contour_orientation.append([orientation,get_box(c,'simple_any')])
        else:
            contour_link[i]['uplink']=-1
            contour_link[i]['downlink']=[]
            contour_orientation.append([orientation,get_box(c,'simple_any')])

    for i,contour in enumerate(contour_orientation):
        if contour[0] == 'counterclockwise':
            min_box = 1000000000
            # 寻找最小包围该内轮廓的外轮廓
            for j,c in enumerate(contour_orientation):
                if c[0] == 'clockwise' and box1_in_box2(contour[1],c[1]) and c[1]['area'] < min_box:
                    min_box = c[1]['area']
                    contour_link[i]['uplink'] = j
            contour_link[contour_link[i]['uplink']]['downlink'].append(i)

    
    return contour_link

def rasterize_cubic(contours, upm=1000, size=(500, 500)):
    if not contours:
        return np.zeros(size)
    h, w = size
    coord = lambda pt: (pt['x']/upm*w, pt['y']/upm*h)
    surface = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    ctx = cairo.Context(surface)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_line_width(10)
    ctx.set_source_rgb(1, 1, 1)
    # Draw the contours
    for contour in contours:
        x0, y0 = coord(contour[0])
        ctx.move_to(x0, y0)
        i = 0
        while i < len(contour) - 1:
            if not contour[i]['on']: raise Exception('Invalid cubic bezier.')
            if not contour[i+1]['on']:
                end_pt = contour[i+3] if i+3 < len(contour) else contour[0]
                end_pt_x, end_pt_y = coord(end_pt)
                x1, y1 = coord(contour[i+1])
                x2, y2 = coord(contour[i+2])
                ctx.curve_to(x1, y1, x2, y2, end_pt_x, end_pt_y)
                i += 3
            else:
                end_pt = contour[i+1] if i+1 < len(contour) else contour[0]
                end_pt_x, end_pt_y = coord(end_pt)
                ctx.line_to(end_pt_x, end_pt_y)
                i += 1
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    ctx.fill()
    # Convert to a numpy array
    buf = surface.get_data()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).copy()
    return arr