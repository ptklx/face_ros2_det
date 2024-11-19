#!/root/miniforge3/envs/myenv/bin/python3.10
import ctypes
import time
import numpy as np
from multiprocessing import Process, sharedctypes, Lock
import os
import rospy
from std_msgs.msg import Int32,String
from sensor_msgs.msg import Image
import ros_numpy
from cv_bridge import CvBridge
import cv2
import json
from detect_face_qnn import face_det_infer ,draw_detect_res,customized_filtration_min_max ,getmin_dis_point ,draw_circle_res

# color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 921600)
# depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 307200)

color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 2764800)
depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 921600)

status_mem = sharedctypes.RawArray(ctypes.c_uint8, 2)  ############
result_mem = sharedctypes.RawArray(ctypes.c_int, 5)

# color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(480,640,3)
# depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(480,640)

color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(720,1280,3)
depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(720,1280)
status = np.frombuffer(status_mem, dtype=ctypes.c_uint8)
color_frame_lock = Lock()
depth_frame_lock = Lock()
result = np.frombuffer(result_mem, dtype=ctypes.c_int)


###### 这是新的摄像头导致获取图像不一样
# def color_image_callback(msg):
#     color_image = ros_numpy.numpify(msg)
#     color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("pt_color.png", color_image)
#     color_frame_lock.acquire()
#     color_frame[:400,:,:] = color_image
#     status[0] = 1
#     color_frame_lock.release()

# def depth_image_callback(msg):
#     depth_image = ros_numpy.numpify(msg)
#     np.save("pt_depth.npy",depth_image)
#     depth_frame_lock.acquire()
#     depth_frame[:400,:] = depth_image
#     status[1] = 1
#     depth_frame_lock.release()



def compute_command_callback(msg):  # 1 
    # rospy.loginfo(rospy.get_caller_id() + "I heard command: %s", msg.data)
    print(f"i heard :{msg.data}")
    try :
        data = json.loads(msg.data)
        need_class = data['message']
        result[0]= need_class
    except  ValueError as e:
        print("the detector not json string")

# def class_command_callback(msg):
#     # rospy.loginfo(rospy.get_caller_id() + "I heard command: %s", msg.data)
#     print(f"i heard class :{msg.data}")
#     result[1]= msg.data

def color_image_callback(msg):
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    #cv2.imwrite("pt_color.png", color_image)
    color_frame_lock.acquire()
    color_frame[:] = color_image
    status[0] = 1
    color_frame_lock.release()


def depth_image_callback(msg):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #np.save("pt_depth.npy",depth_image)
    depth_frame_lock.acquire()
    depth_frame[:] = depth_image
    status[1] = 1
    depth_frame_lock.release()


# def start_recording(self):
#     self.update_result_text("准备录音...")  # 在开始录音之前更新状态
#     subprocess.call(['python3', '/home/aidlux/ros_ws/src/my_arm_control_pkg/scripts/test.py'])  # 调用test.py脚本来执行录音

# def start_recording_route(self):
#     threading.Thread(target=self.start_recording).start()  # 在后台线程中启动录音
#     return "Recording started", 200




def center_manager():
    rospy.init_node('face_manager', anonymous=True)
    rospy.Subscriber('/cam_2/color/image_raw', Image, color_image_callback)
    # rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_image_callback)
    rospy.Subscriber('/cam_2/aligned_depth_to_color/image_raw',Image, depth_image_callback)  # 对齐的图像
    # rospy.Subscriber('/camera/rgb/image_color', Image, color_image_callback)
    # rospy.Subscriber('/camera/depth/image', Image, depth_image_callback)

    
    rospy.Subscriber('/robot/arm/commond', String, compute_command_callback)
    # rospy.Subscriber('/robot/voice/commond', Int32, class_command_callback)

    os.environ['ADSP_LIBRARY_PATH'] = '/usr/local/lib/aidlux/aidlite/snpe2/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'

    qnn_model =face_det_infer(os.path.join(os.path.dirname(os.path.abspath(__file__)),'face/8550scrfd_face_fp16.qnn216.ctx.bin'))


    pub = rospy.Publisher('/ai/facedet/result', String, queue_size=1)
    # pub_starter = rospy.Publisher('/ai/detecter/start', String, queue_size=1)
    rate = rospy.Rate(15)
    time_st = 20  # 延迟1s多
    st =0
    result_str = json.dumps({'to': 'llm', 'message': 'no'})
    max_leng = 1500
    start_leng = 1000
    tmp_distance =2000
    first_flag =True
    result[0]=0  # test
    result[2]=0  # inter
    while not rospy.is_shutdown():
        if result[2] != 1:
            # print("start detect !")
            ## get image
            color_frame_lock.acquire()
            frame = color_frame.copy()
            status[0] = 0
            color_frame_lock.release()
            depth_frame_lock.acquire()
            dframe = depth_frame.copy()
            status[1] = 0
            depth_frame_lock.release()
            ## det
            preds = qnn_model.predict_det(frame)
            
            if len(preds):
                showframe = draw_detect_res(frame, preds)   ### 测试
                det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
                if len(det_pred):
                    one_face_info =  getmin_dis_point(det_pred,dframe)
                    showframe = draw_circle_res(showframe,det_pred[one_face_info[0]])
                    print(f"the min distance: {one_face_info[1]}")
                    # result_str = "{}".format(int(one_face_info[1]))  ############
                    if one_face_info[1]<start_leng and first_flag:
                        first_flag=False
                        # result_str = result_str.replace('no',"sayhello")
                        result_str = json.dumps({'to': 'llm', 'message': 'sayhello'})
                        pub.publish(result_str)
                        rospy.loginfo("Published: %s", result_str)
                        result[2]=1
                  
                        tmp_distance= one_face_info[1]
                    # elif result[2]!=1:
                    #     result_str = '''{"to": "llm", "message": 'no'}'''
                    #     pub.publish(result_str)
                    if one_face_info[1]>max_leng:
                        first_flag=True

                else:
                    first_flag=True
                    print("too close or too far")
                cv2.imwrite("result.jpg", showframe)
                print(result_str)
        else:
            # result_str = '''{"to": "llm", "message": 'no'}'''
            # print("not detect object")
        ###
        # if result[0] != -1 and st==0:
        #     result[2]=1
        #     result_str = result_str.replace('no',"sayhello")
        #     pub.publish(result_str)
        #     print("inter ")
        #     result[0] = -1
        # else:
            if result[2] !=0:  # 用于延迟和规避掉一直sayhello
                result_str = json.dumps({'to': 'llm', 'message': 'no'})
                if st < time_st:
                    st += 1
                if st >= time_st:
                    pub.publish(result_str)
                    result[2] = 0
                 
                    st = 0
        rate.sleep()
            
        
            
        
if __name__ == "__main__":
    try:
        center_manager()
    except rospy.ROSInterruptException:
        pass