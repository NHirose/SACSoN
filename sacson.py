#!/usr/bin/env python

import os

#ROS
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

#NN model
from network import Sacson_net

#torch
import torch
import torch.nn.functional as F

#others
import cv2
from cv_bridge import CvBridge
import numpy as np
import pickle 
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--robot_radius", type=float, help="robot radius", default=0.5)
args = parser.parse_args()

print("Inference with ROS!")

#flag for initail step
init_hist = 0

#loading model parameters
model_file_sacson = os.path.join(os.path.dirname(__file__), 'models/polinet.pth')

# resize parameters
rsizex = 128
rsizey = 128

#mask for 360 degree image
mask_br360 = np.loadtxt(open(os.path.join(os.path.dirname(__file__), "utils/mask_360view.csv"), "rb"), delimiter=",", skiprows=0)
#print mask_br.shape
mask_brr = mask_br360.reshape((1,1,128,256)).astype(np.float32)
mask_brr1 = mask_br360.reshape((1,128,256)).astype(np.float32)
mask_brrc = np.concatenate((mask_brr1, mask_brr1, mask_brr1), axis=0)

def unddp_state_dict(state_dict):
    if not all([s.startswith("module.") for s in state_dict.keys()]):
        return state_dict

    return OrderedDict((k[7:], v) for k, v in state_dict.items())

def preprocess_image(msg):
    cv_img = bridge.imgmsg_to_cv2(msg)
    cv_resize_n = cv2.resize(cv_img, (2*rsizex, rsizey), cv2.INTER_AREA)
    cv_resizex = cv_resize_n.transpose(2, 0, 1)
    in_imgcc1 = np.array([cv_resizex], dtype=np.float32)
    in_img1 = (in_imgcc1 - 127.5)/127.5

    img_nn_cL = mask_brrc * (in_img1 + 1.0) -1.0 #mask
    img = img_nn_cL.astype(np.float32)

    return img

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

def callback_360(msg_1):
    global init_hist    
    global xhist1, xhist2, xhist3, xhist4, xhist5, xhist6    

    if True:
        cur_img = preprocess_image(msg_1) #current image
        cur_img_360 = cur_img
        goal_img_360 = goal_img

        #normalization betweeen current and subgoal image
        imgrc = (np.reshape(cur_img[0][0],(1,128,256)) + 1.0)*0.5
        imggc = (np.reshape(cur_img[0][1],(1,128,256)) + 1.0)*0.5
        imgbc = (np.reshape(cur_img[0][2],(1,128,256)) + 1.0)*0.5
        mean_cbgr = np.zeros((3,1))
        std_cbgr = np.zeros((3,1))
        mean_ct = np.zeros((3,1))
        std_ct = np.zeros((3,1))
        mean_cbgr[0] = np.sum(imgrc)/countm
        mean_cbgr[1] = np.sum(imggc)/countm
        mean_cbgr[2] = np.sum(imgbc)/countm
        std_cbgr[0] = np.sqrt(np.sum(np.square(imgrc-mask_brr1*mean_cbgr[0]))/countm)
        std_cbgr[1] = np.sqrt(np.sum(np.square(imggc-mask_brr1*mean_cbgr[1]))/countm)
        std_cbgr[2] = np.sqrt(np.sum(np.square(imgbc-mask_brr1*mean_cbgr[2]))/countm)

        imgrt = (np.reshape(goal_img[0][0],(1,128,256)) + 1)*0.5
        imggt = (np.reshape(goal_img[0][1],(1,128,256)) + 1)*0.5
        imgbt = (np.reshape(goal_img[0][2],(1,128,256)) + 1)*0.5
        mean_tbgr = np.zeros((3,1))
        std_tbgr = np.zeros((3,1))
        mean_tt = np.zeros((3,1))
        std_tt = np.zeros((3,1))
        mean_tbgr[0] = np.sum(imgrt)/countm
        mean_tbgr[1] = np.sum(imggt)/countm
        mean_tbgr[2] = np.sum(imgbt)/countm
        std_tbgr[0] = np.sqrt(np.sum(np.square(imgrt-mask_brr1*mean_tbgr[0]))/countm)
        std_tbgr[1] = np.sqrt(np.sum(np.square(imggt-mask_brr1*mean_tbgr[1]))/countm)
        std_tbgr[2] = np.sqrt(np.sum(np.square(imgbt-mask_brr1*mean_tbgr[2]))/countm)

        mean_tt[0] = mean_tbgr[0]
        mean_tt[1] = mean_tbgr[1]
        mean_tt[2] = mean_tbgr[2]
        std_tt[0] = std_tbgr[0]
        std_tt[1] = std_tbgr[1]
        std_tt[2] = std_tbgr[2]
        mean_ct[0] = mean_cbgr[0]
        mean_ct[1] = mean_cbgr[1]
        mean_ct[2] = mean_cbgr[2]
        std_ct[0] = std_cbgr[0]
        std_ct[1] = std_cbgr[1]
        std_ct[2] = std_cbgr[2]

        xcg = torch.clamp(torch.from_numpy(cur_img).to(device), -1.0, 1.0)
        xcg_360 = torch.clamp(torch.from_numpy(cur_img_360).to(device), -1.0, 1.0)
        xcgf_360 = xcg_360[:, :, :, 0:rsizex]
        xcgb_360 = xcg_360[:, :, :, rsizex:2*rsizex]#.flip(3)
        xcg_360 = torch.cat((xcgf_360, xcgb_360),axis=1)

        imgrcc = (imgrc-mean_cbgr[0])/std_cbgr[0]*std_tt[0]+mean_tt[0]
        imggcc = (imggc-mean_cbgr[1])/std_cbgr[1]*std_tt[1]+mean_tt[1]
        imgbcc = (imgbc-mean_cbgr[2])/std_cbgr[2]*std_tt[2]+mean_tt[2]
        curc_img = np.array((np.reshape(np.concatenate((mask_brr*imgrcc, mask_brr*imggcc, mask_brr*imgbcc), axis = 0), (1,3,128,256))*mask_c - 0.5)*2.0, dtype=np.float32)
        cimage = torch.clamp(torch.from_numpy(curc_img).to(device), -1.0, 1.0) #normalized current image


        imgrtt = (imgrt-mean_tbgr[0])/std_tbgr[0]*std_ct[0]+mean_ct[0]
        imggtt = (imggt-mean_tbgr[1])/std_tbgr[1]*std_ct[1]+mean_ct[1]
        imgbtt = (imgbt-mean_tbgr[2])/std_tbgr[2]*std_ct[2]+mean_ct[2]
        goalt_img = np.array((np.reshape(np.concatenate((mask_brr*imgrt, mask_brr*imggt, mask_brr*imgbt), axis = 0), (1,3,128,256)) - 0.5)*2.0, dtype=np.float32)
        timage = torch.clamp(torch.from_numpy(goalt_img).to(device), -1.0, 1.0) #normalized goal image

        #current image
        xcgf = cimage[:, :, :, 0:rsizex]
        xcgb = cimage[:, :, :, rsizex:2*rsizex]
        xcgx = torch.cat((xcgf.flip(1), xcgb.flip(1)),axis=1) #6ch current image 

        #subgoal image
        xpgf = timage[:, :, :, 0:rsizex]
        xpgb = timage[:, :, :, rsizex:2*rsizex]
        xpgx = torch.cat((xpgf.flip(1), xpgb.flip(1)),axis=1) #6ch subgoal image 

        #only at the first step, fill the past obervations by the current image.
        """
        try:
            image_cat = torch.cat((xcgx, xhist1, xhist2, xhist3, xhist4, xhist5, xhist6, xpgx), axis=1)
            print("try")            
        except:
            xhist6 = xcgx
            xhist5 = xcgx
            xhist4 = xcgx
            xhist3 = xcgx
            xhist2 = xcgx
            xhist1 = xcgx
            image_cat = torch.cat((xcgx, xhist1, xhist2, xhist3, xhist4, xhist5, xhist6, xpgx), axis=1)
            print("except")      
        """
        #print(xhist1, xhist2, xhist3, xhist4, xhist5, xhist6)    
          
        
        if init_hist == 0:
            xhist6 = xcgx
            xhist5 = xcgx
            xhist4 = xcgx
            xhist3 = xcgx
            xhist2 = xcgx
            xhist1 = xcgx
            init_hist = 1

        #condanatenated images with current, history and subgoal images
        image_cat = torch.cat((xcgx, xhist1, xhist2, xhist3, xhist4, xhist5, xhist6, xpgx), axis=1)
        
        robot_size = args.robot_radius*torch.ones(1, 1, 1, 1).to(device) #robot radius for generating velocity commands
        robot_size_te = args.robot_radius*torch.ones(1, 1, 1, 1).to(device) #robot radius for traversability estimation
               
        #setting initial position on local robot coordinate
        px = torch.zeros(1).to(device)
        pz = torch.zeros(1).to(device)
        ry = torch.zeros(1).to(device)

        with torch.no_grad():
            vwres, ptrav = polinet(image_cat, robot_size, robot_size_te, px, pz, ry)

        vt = vwres.cpu().numpy()[0,0,0,0] #linear velocity to control the robot
        wt = vwres.cpu().numpy()[0,1,0,0] #angular velocity to control the robot

        print("Linear vel.:", vt, "Angular vel.:", wt)
        
        #publish velocity command
        msg_pub = Twist()
        msg_pub.linear.x = vt
        msg_pub.linear.y = 0.0
        msg_pub.linear.z = 0.0
        msg_pub.angular.x = 0.0
        msg_pub.angular.y = 0.0
        msg_pub.angular.z = wt
        
        msg_out.publish(msg_pub)
        
        #updating history of images
        xhist6 = xhist5
        xhist5 = xhist4
        xhist4 = xhist3
        xhist3 = xhist2
        xhist2 = xhist1
        xhist1 = xcgx

def callback_ref(msg):
    global goal_img
    goal_img = preprocess_image(msg) #subgoal image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#network definition for SACSoN
dof_size = 2
step_size = 8
lower_bounds = [0.0, -1.0]
upper_bounds = [+0.5, +1.0]

polinet = Sacson_net(48, step_size, lower_bounds, upper_bounds).to(device) #image_history
polinet.load_state_dict(unddp_state_dict(torch.load(model_file_sacson, map_location=device)))
polinet.eval()

bridge = CvBridge()

goal_img = np.zeros((1,3,128,256), dtype=np.float32)

countm = 0
for it in range(128):
    for jt in range(256):
        if mask_brr[0][0][it][jt] > 0.5:
            countm += 1

mask_c = np.concatenate((mask_brr, mask_brr, mask_brr), axis=1)


# main function
if __name__ == '__main__':

    #initialize node
    rospy.init_node('SACSoN', anonymous=False)
    
    #subscriber of topics
    msg1_sub = rospy.Subscriber('/topic_name_current_image', Image, callback_360, queue_size=1) #current observation from Ricoh Theta
    msg2_sub = rospy.Subscriber('/topic_name_goal_image', Image, callback_ref, queue_size=1) #subgoal image
        
    #publisher of topics
    msg_out = rospy.Publisher('/cmd_vel', Twist, queue_size=1) #velocities for the robot control

    print('waiting message .....')
    
    rospy.spin()
