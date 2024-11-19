import os
import random
import math
import time
import onnxruntime as ort
import numpy as np
import cv2
from math import ceil
from itertools import product 
import config  



def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = None
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = None
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                    priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                    ), axis=1)

    return landms


class PriorBox(object):
    def __init__(self, cfg, format:str="tensor", image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
        self.__format = format

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
       
        if self.clip:
            if self.__format == "tensor":
                output.clamp_(max=1, min=0)
            else:
                output = np.clip(output, 0, 1)

        return output



class face_det_infer():
    def __init__(self,path,im_height, im_width):
        det_model_path = path#os.path.join(os.path.dirname(os.path.abspath(__file__)), "./models/m_faceDetctor.onnx")
        self.det = ort.InferenceSession(det_model_path,providers=['CPUExecutionProvider'])
        self.input_det_name = [input.name for input in self.det.get_inputs()]
        self.cfg = config.cfg_mnet
        self.resize = 1
        self.vis_thres =0.65
        self.confidence_threshold=0.68
        self.top_k= 100
        self.keep_top_k = 50
        self.nms_threshold = 0.2
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width), format="numpy")
        priors = priorbox.forward()
        self.prior_data = priors



    def predict_det(self,image):
        tic = time.time()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.float32(img_rgb)
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        loc, conf, landms = self.det.run(None,{self.input_det_name[0]:img})

        boxes = decode(np.squeeze(loc, axis=0), self.prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        scores = np.squeeze(conf, axis=0)[:, 1]

        landms = decode_landm(np.squeeze(landms.data, axis=0), self.prior_data, self.cfg['variance'])

        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        landms = landms * scale1 / self.resize

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
       
        # print('net cost time: {:.4f}s'.format(time.time() - tic))
        return dets
    

    def draw_detect_res(self,img_raw, dets):
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4) # 红色，左眼
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4) #鼻子
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)  # 绿左嘴角
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4) # 蓝色，右嘴角
        return img_raw

    def is_front_face(self, keypoints: np.ndarray): # -> Tuple[List[float], bool, List[float]]:
        '''
        侧脸判断
        抬头仰头判断

        Args:
            keypoints: 人脸关键点(左眼、右眼、鼻子、左嘴角、右嘴角)

        Returns: orientation, isfront
        '''
        keypoints=keypoints.reshape(5,2)   ####
        isfront = True
        orientation = []

        # 侧脸判断
        a = self.orthogonal_point(keypoints[0], keypoints[1], keypoints[2])
        m = (keypoints[0] + keypoints[1]) / 2
        dis_front = np.linalg.norm(a - m) / np.maximum(np.linalg.norm(keypoints[0] - keypoints[1]), 1e-10)
        orientation.append(dis_front)

        # 抬头低头判断 取交点，满足：$(x_3 -a) = \alpha(b - a)$
        k1, c1 = self.linear_equation(keypoints[2][0], keypoints[2][1], a[0], a[1])
        k2, c2 = self.linear_equation(keypoints[3][0], keypoints[3][1], keypoints[4][0], keypoints[4][1])
        b = self.linear_equation(k1, c1, k2, c2, beg_k=False)  # x3a与x4x5的交点
        dis_bow = np.linalg.norm(keypoints[2] - a) / np.maximum(np.linalg.norm(b - a), 1e-10)
        orientation.append(dis_bow)

        # >0.5算侧脸、[0.2, 0.7]之外算仰头低头
        if (dis_front > 0.5) or (not 0.2 < dis_bow < 0.7):
            isfront = False

        # print("[DEBUG] orientation: ", orientation)

        return orientation, isfront, [a, b]

    def orthogonal_point(self, x1, x2, x3):
        '''
        已知：(x1,y1)、(x2,y2)、(x3,y3)
        求：a = (ax, ay)
        ∵ 向量x1x2 = (x2-x1,y2-y1) 正交于 向量ax3 = (x3-ax, y3-ay)
        ∴ (x2 - x1)(x3 - ax) + (y2-y1)(y3-ay) = 0
        ∵ a在向量x1x2上, y = λx + b
        ∴ (x2 - x1)(x3 - ax) + (y2-y1)(y3- λax + b) = 0
        ∴ ax = (x3*((x2 - x1)) + y3*(y2-y1) - (y2-y1)*b) / (((x2 - x1)) + (y2-y1)*λ)
        ∴ ay = λ ax + b
        '''
        x1, y1 = x1[0], x1[1]
        x2, y2 = x2[0], x2[1]
        x3, y3 = x3[0], x3[1]
        # conb, lambdax = np.linalg.solve(np.array([[1, x1], [1, x2]]), np.array([y1, y2]))
        k, b = self.linear_equation(x1, y1, x2, y2, beg_k=True)
        ax = (x3 * (x2 - x1) + y3 * (y2 - y1) - (y2 - y1) * b) / np.maximum((x2 - x1) + (y2 - y1) * k, 1e-10)
        ay = k * ax + b

        return np.array([ax, ay])

    def linear_equation(self, x1, y1, x2, y2, beg_k=True):
        '''
        + beg_k=True：两点确定一条直线
        + beg_k=False: 求两条直线的交点
            a, x3, x4, x5 -> b
            直线1：y = k1 * x + c1
            直线2：y = k2 * x + c2
            x = (c1 - c2) / (k2 - k1)
            y = k1 * x + c1
        Args:
            beg_k: true 为解方程求截取和斜率，传入为x1, y1, x2, y2。false 为求x和y，传入为两个方程的截距和斜率

        Returns: (k, b) or (x, y)
        '''
        if beg_k:
            k = (y2 - y1) / np.maximum((x2 - x1), 1e-10)
            b = y1 - k * x1
            return k, b
        else:
            k1, c1, k2, c2 = x1, y1, x2, y2
            x = (c1 - c2) / np.maximum((k2 - k1), 1e-10)
            y = k1 * x + c1
            return np.array([x, y])


####

def customized_filtration_min_max(img, det_pred):
    middle_w = img.shape[1]/2   # 0.3宽，
    middle_h = img.shape[0]/2   # 0.3高
    dels = []
    for i in range(len(det_pred)):
        x, y, x2, y2 = [int(t) for t in det_pred[i][:4]]
        w = x2-x
        h = y2-y
        if w>middle_w or h >middle_h:  # 排除过大的框 
            dels.append(i)
        if  w<middle_w/20 or h <middle_h/20: # 排除过小的框
            dels.append(i)
        if x <8 or x>img.shape[1]-8:
            dels.append(i)
        if y <8  or y>img.shape[0]-8:
            dels.append(i)
        
    det_pred = np.delete(det_pred, dels, axis=0)
    return det_pred

def get_mid_pos(box,depth_data,randnum):
    distance_list = []
    # mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    mid_pos = [box[0],box[1]]
    h,w = depth_data.shape
    # min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    min_val=min(abs(box[2]-4),abs(box[3]-4))
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        y_p = int(mid_pos[1] + bias)
        x_p = int(mid_pos[0] + bias)
        y_p = y_p if y_p>0 else 0
        y_p = h-1 if y_p>h-1 else y_p

        x_p = x_p if x_p>0 else 0
        x_p = w-1 if x_p>w-1 else x_p

        dist = depth_data[y_p, x_p]
        # cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    # print("distance_list:",distance_list)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    dis_mean = np.mean(distance_list)
    # print("dis_mean",dis_mean)
    if math.isnan(dis_mean):
        return 100000
    else:
        return int(dis_mean)

def getmin_dis_point(det_pred,dframe):
    center_dis = []
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        w = x2-x1
        h = y2-y1
        center_info = [(x1+x2)/2,(y1+y2)/2,w,h]
        dist = get_mid_pos(center_info,dframe,24)
        center_dis.append(dist)
    if center_dis==[]:
        return center_dis
    mindis = min(center_dis)
    my_index = center_dis.index(mindis)
    return [my_index,mindis]

def draw_circle_res(img,point):
    radius=10
    color = (0,0,255)
    thickness=-1
    cv2.circle(img, (int((point[0]+point[2])/2),int((point[1]+point[3])/2)), radius, color, thickness)
    return img




def main_pic():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/FaceDetector_dy.onnx")
    image_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"models/liu.png")
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    im_height, im_width, _ = img_raw.shape
    retinaface =face_det_infer(path,im_height, im_width)

    dets = retinaface.predict_det(img_raw)

    for one_det in dets:
        key_point = one_det[5:]
        re_info = retinaface.is_front_face(key_point)
        if re_info[1]:
            print("this is front face ok !")
        else:
            print("not !")

    if len(dets):
        showframe = retinaface.draw_detect_res(img_raw,dets)
        name = "test.jpg"
        cv2.imwrite(name, img_raw)
        # cv2.imshow("test",showframe)
        # cv2.waitKey(0)
        det_pred = customized_filtration_min_max(img_raw, dets)   # 删除不在范围中的

        if len(det_pred):
            # one_face_info =  getmin_dis_point(det_pred,dframe)
            # showframe = draw_circle_res(showframe,det_pred[one_face_info[0]])
            # print(f"the min distance: {one_face_info[1]}")
            pass
        else:
            print("too close or too far")

        cv2.imwrite("result_onnx.jpg", showframe)


    
if __name__ == "__main__":
    main_pic()
