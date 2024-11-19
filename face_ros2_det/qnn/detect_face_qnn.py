import time
import numpy as np
import cv2
import aidlite
import random
# import argparse
import os



def eqprocess(image,size1,size2):
    h,w,_ = image.shape
    mask = np.zeros((size1,size2,3),dtype=np.float32)
    scale1 = h /size1
    scale2 = w / size2
    if scale1 > scale2:
        scale = scale1
    else:
        scale = scale2
    img = cv2.resize(image,(int(w / scale),int(h / scale)))
    mask[:int(h / scale),:int(w / scale),:] = img
    return mask,scale

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms(dets, iou_thres):
    thresh = iou_thres
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

class face_det_infer():
    def __init__(self,path):
        # current_p =os.path.dirname(os.path.abspath(__file__))
        dpath = path #os.path.join(current_p,"./face/scrfd_face_w8a16.qnn216.ctx.bin")
        self.target_model = dpath #os.path.join(current_p,'./models/cutoff_yolov5n_w8a8.qnn216.ctx.bin')
        self.model_type = 'QNN'
        self.init_qnn()

    def init_qnn(self):
        config = aidlite.Config.create_instance()
        if config is None:
            print("Create config failed !")
            return False
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if self.model_type.lower()=="qnn":
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        elif self.model_type.lower()=="snpe2" or self.model_type.lower()=="snpe":
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
            
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1
        
        model = aidlite.Model.create_instance(self.target_model)
        if model is None:
            print("Create model failed !")
            return False
        input_shapes = [[1, 480, 640, 3]]
        self.output_shapes = [[9600,1],[9600,4],[9600,10], [2400,1], [2400,4],[2400,10],[600,1],[600,4],[600,10]]
        model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                                self.output_shapes, aidlite.DataType.TYPE_FLOAT32)

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if  self.interpreter is None:
            print("build_interpretper_from_model_and_config failed !")
            return None
        result =  self.interpreter.init()
        if result != 0:
            print(f"interpreter init failed !")
            return False
        result =  self.interpreter.load_model()
        if result != 0:
            print("interpreter load model failed !")
            return False
        print("detect model load success!")
   
    def qnn_pre(self,data):
        result =  self.interpreter.set_input_tensor(0, data.data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
        
        t1=time.time()
        result =  self.interpreter.invoke()
        cost_time = (time.time()-t1)*1000
        if result != 0:
            print("interpreter set_input_tensor() failed")

        stride0 =  self.interpreter.get_output_tensor(0)
        stride1 =  self.interpreter.get_output_tensor(1)
        stride2 =  self.interpreter.get_output_tensor(2)
        stride3 =  self.interpreter.get_output_tensor(3)
        stride4 =  self.interpreter.get_output_tensor(4)
        stride5 =  self.interpreter.get_output_tensor(5)
        stride6 =  self.interpreter.get_output_tensor(6)
        stride7 =  self.interpreter.get_output_tensor(7)
        stride8 =  self.interpreter.get_output_tensor(8)

        print("=======================================")
        print(f"inference time {cost_time}")
        print("=======================================")


        validCount0 = stride0.reshape(*self.output_shapes[0])
        validCount1 = stride1.reshape(*self.output_shapes[1])#.transpose(1, 0)
        validCount2 = stride2.reshape(*self.output_shapes[2])#.transpose(1, 0)
        validCount3 = stride3.reshape(*self.output_shapes[3])
        validCount4 = stride4.reshape(*self.output_shapes[4])#.transpose(1, 0)
        validCount5 = stride5.reshape(*self.output_shapes[5])#.transpose(1, 0)
        validCount6 = stride6.reshape(*self.output_shapes[6])
        validCount7 = stride7.reshape(*self.output_shapes[7])#.transpose(1, 0)
        validCount8 = stride8.reshape(*self.output_shapes[8])#.transpose(1, 0)
        return[validCount0,validCount3,validCount6,validCount1,validCount4,validCount7,validCount2,validCount5,validCount8]

    def destory(self):
        result =  self.interpreter.destory()


    def predict_det(self,image):
        image, scale = eqprocess(image, 480, 640)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # input = ((image - 127.5) / 128).transpose(2,0,1)[None].astype(np.float32)
        input = ((image - 127.5) / 128)[None].astype(np.float32)
        feat_stride_fpn = [8, 16, 32]
        input_height = 480
        input_width = 640
        conf_thres = 0.6
        iou_thres = 0.4
        
        center_cache = {}
        # net_outs = []
        scores_list = []
        bboxes_list = []

        
        # outputs = self.det.run(None,{"input.1":input})
        outputs= self.qnn_pre(input)
        
        for idx, stride in enumerate(feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + 3]
            bbox_preds = bbox_preds * stride
            
            height = input_height // stride
            width = input_width // stride
            
            key = (height, width, stride)
            
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
            
            if len(center_cache)<100:
                center_cache[key] = anchor_centers
                        
            pos_inds = np.where(scores >= conf_thres)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) * scale
            
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det, iou_thres)
        det = pre_det[keep, :]
        return det
    
    def predict_rec(self,crop_frame):
        image = cv2.resize(crop_frame, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input = (image / 255).transpose(2,0,1)[None].astype(np.float32)
        output = self.rec.run(None,{"input.1":input})[0]

        return output
    
def draw_detect_res(img, det_pred):
    img = img.astype(np.uint8)
    color_step = int(255 / 1)
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        print(i + 1, [x1, y1, x2, y2], score)
        cv2.rectangle(img, (x1, y1), (x2 , y2 ), (0, int( color_step), int(255 - color_step)),
                      thickness=2)
    return img

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
    # dframe = np.load(os.path.join('./',"pt_depth.npy"))
    path = os.path.dirname(os.path.abspath(__file__))
    face_path = os.path.join(path,"./face/8550scrfd_face_fp16.qnn216.ctx.bin")
    onnx_face = face_det_infer(face_path)
    frame = cv2.imread( os.path.join(os.path.dirname(os.path.abspath(__file__)),"face/liu.png"))
    frame = cv2.resize(frame,(640,480))
    # img_processed = np.copy(frame)
    det = onnx_face.predict_det(frame)
    if len(det):
        showframe = draw_detect_res(frame, det)   ### 测试
        det_pred = customized_filtration_min_max(frame, det)   # 删除不在范围中的
        if len(det_pred):
            # one_face_info =  getmin_dis_point(det_pred,dframe)
            # showframe = draw_circle_res(showframe,det_pred[one_face_info[0]])
            # print(f"the min distance: {one_face_info[1]}")
            pass
        else:
            print("too close or too far")

        cv2.imwrite("result.jpg", showframe)
    onnx_face.destory()

    
if __name__ == "__main__":
    main_pic()

