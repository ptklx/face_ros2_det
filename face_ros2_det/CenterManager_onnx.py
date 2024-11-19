import ctypes
import numpy as np
import time
from multiprocessing import sharedctypes, Lock
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32,String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import json

from detect_face_onnx import face_det_infer ,customized_filtration_min_max ,getmin_dis_point ,draw_circle_res

# os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'

#os.environ['ADSP_LIBRARY_PATH'] = '/usr/local/lib/aidlux/aidlite/snpe2/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
class FaceManager(Node):
    def __init__(self):
        super().__init__('yolo_manager')
        self.get_logger().info("start face detect!!!")
        color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 921600)
        depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 307200)

        # color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 2764800)
        # depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 921600)

        status_mem = sharedctypes.RawArray(ctypes.c_uint8, 2)  ############
        result_mem = sharedctypes.RawArray(ctypes.c_int, 5)
        im_height=480   #720
        im_width=640    #1280
        # self.color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(480,640,3)
        # self.depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(480,640)
        #
        self.color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(im_height,im_width,3)
        self.depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(im_height,im_width)

        self.status = np.frombuffer(status_mem, dtype=ctypes.c_uint8)
        self.color_frame_lock = Lock()
        self.depth_frame_lock = Lock()
        self.result = np.frombuffer(result_mem, dtype=ctypes.c_int)

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models/FaceDetector_dy.onnx')
        self.qnn_model =face_det_infer(model_path,im_height, im_width)


        self.subscription_color = self.create_subscription(
            Image,
            '/cam_2/cam_2/color/image_raw',
            self.color_image_callback,
            10
        )
        self.subscription_depth = self.create_subscription(
            Image,
            '/cam_2/cam_2/aligned_depth_to_color/image_raw',
            self.depth_image_callback,
            10
        )
        self.subscription_command = self.create_subscription(
            String,
            '/robot/arm/commond',
            self.compute_command_callback,
            10
        )
        self.publisher_result = self.create_publisher(String, '/ai/facedet/result', 10)
        self.time_st = 10  # 延迟1s多   由于检测耗时，也就设置短
        self.last_time = 5
        self.st =0
        self.result_str = json.dumps({'to': 'llm', 'message': 'no'})
        self.max_leng = 1500
        self.start_leng = 1000
        self.last_execution_time = time.time()
        self.first_flag =True
        self.result[0]=0  #  获取二次触发命令
        self.result[2]=0  # inter

        self.timer = self.create_timer(1.0 /30.0, self.timer_callback)

    def color_image_callback(self, msg):
        try:
            bridge = CvBridge()
            color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            #cv2.imwrite("pt_color.png", color_image)
            self.color_frame_lock.acquire()
            self.color_frame[:] = color_image
            self.status[0] = 1
            self.color_frame_lock.release()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        else:
            cv2.imwrite("pt_color.png", color_image)
            pass

    def depth_image_callback(self, msg):
        try:
            bridge = CvBridge()
            # 将 ROS 图像消息转换为 OpenCV 图像
            # cv_image = bridge.imgmsg_to_cv2(msg, "16UC1")
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_frame_lock.acquire()
            self.depth_frame[:] = depth_image
            self.status[1] = 1
            self.depth_frame_lock.release()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        else:
            # 在这里处理深度图像
            pass


    def compute_command_callback(self, msg):
        # self.get_logger().info(f"Received command: {msg.data}")
        print(f"i heard :{msg.data}")
        try :
            data = json.loads(msg.data)
            need_class = data['message']
            self.result[0]= need_class
        except   CvBridgeError as e:
            print("the detector not json string")


    def timer_callback(self):
        if self.result[2] != 1:
            # print("start detect !")
            ## get image
            self.color_frame_lock.acquire()
            frame = self.color_frame.copy()
            self.status[0] = 0
            self.color_frame_lock.release()
            self.depth_frame_lock.acquire()
            dframe = self.depth_frame.copy()
            self.status[1] = 0
            self.depth_frame_lock.release()
            ## det
            preds = self.qnn_model.predict_det(frame)
            
            if len(preds):
                showframe = self.qnn_model.draw_detect_res(frame, preds)   ### 测试
                det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
             
                front_face =[]
                for one_det in det_pred:
                    key_point = one_det[5:]
                    re_info = self.qnn_model.is_front_face(key_point)
                    if re_info[1]:
                        front_face.append(one_det)
                        print("this is front face ok !")
                    else:
                        print("not !")
                if front_face!=[]:
                    one_face_info =  getmin_dis_point(front_face,dframe)
                    if one_face_info!=[]:
                        print(f"the min distance: {one_face_info[1]}")
                        showframe = draw_circle_res(showframe,front_face[one_face_info[0]])  #测试
                        if one_face_info[1]<self.start_leng and self.first_flag:
                            self.first_flag=False
                            self.result_str = json.dumps({'to': 'llm', 'message': 'sayhello'})
                            msg = String()
                            msg.data = self.result_str
                            self.publisher_result.publish(msg)
                            
                            self.get_logger().info(f"Received command: {self.result_str}")
                            self.result[2]=1
                    
                        current_time = time.time()
                        # 计算从上次执行到现在的时间间隔
                        if current_time - self.last_execution_time >= self.last_time and one_face_info[1]>self.max_leng:
                            self.first_flag=True
                            self.last_execution_time = current_time
                            print(f"restart face flag !!!!!!!")
                        # 通过延时触发
                        # if  current_time - self.last_execution_time >= 30 and one_face_info[1]<self.start_leng: # 30s 半分钟后重启
                        #     self.first_flag=True
                        #     self.last_execution_time = current_time
                        #     print(f"too long time restart face flag !!!!!!!")
                        # 通过命令触发
                        if  current_time - self.last_execution_time >= 10 and one_face_info[1]<self.max_leng and self.result[0]==1:
                            self.result[0]=0
                            self.first_flag=True
                            self.last_execution_time = current_time
                            print(f"too long time restart face flag !!!!!!!")

                    else:
                        print("det is null")

                cv2.imwrite("result.jpg", showframe)
                print(self.result_str)
        else:
            if self.result[2] !=0:  # 用于延迟和规避掉一直sayhello
                self.result_str = json.dumps({'to': 'llm', 'message': 'no'})
                if self.st < self.time_st:
                    self.st += 1
                if self.st >= self.time_st:
                    msg = String()
                    msg.data = self.result_str
                    self.publisher_result.publish(msg)
                    self.result[2] = 0
                    self.st = 0



def center_manager():
    rclpy.init()
    node = FaceManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == "__main__":
    try:
        center_manager()
    except Exception as e:
        print(f"Exception occurred :{e}")