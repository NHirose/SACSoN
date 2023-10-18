#!/usr/bin/env python

import os

#NN model
from network import Sacson_net
from pednet import PedNet

#torch
import torch
import torch.nn.functional as F

#others
import cv2
import numpy as np
import pickle 
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--robot_radius", type=float, help="robot radius", default=0.5)
parser.add_argument("--sample", action='store_true', help="visualization using sample images")
args = parser.parse_args()

print("Visualization using sample images!")

def unddp_state_dict(state_dict):
    if not all([s.startswith("module.") for s in state_dict.keys()]):
        return state_dict

    return OrderedDict((k[7:], v) for k, v in state_dict.items())

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#network definition for SACSoN
dof_size = 2
step_size = 8
lower_bounds = [0.0, -1.0]
upper_bounds = [+0.5, +1.0]

#loading parameters of sacson
model_file_sacson = os.path.join(os.path.dirname(__file__), 'models/polinet.pth')
polinet = Sacson_net(48, step_size, lower_bounds, upper_bounds).eval().to(device)
polinet.load_state_dict(unddp_state_dict(torch.load(model_file_sacson, map_location=device)))
polinet

#loading parameters of pedestrians' dynamic forward model
model_file_pednet = os.path.join(os.path.dirname(__file__), 'models/pednet.pth')
pednet = PedNet(8, 8).eval().to(device) 
pednet.load_state_dict(unddp_state_dict(torch.load(model_file_pednet, map_location=device)))

# main function
if __name__ == '__main__':

    #loading sample data
    with open( './sample/sample.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    #normalization of images
    cur_i = ((torch.from_numpy(loaded_dict["cur_img"]).float() - 127.5)/127.5).to(device).unsqueeze(0)
    goal_i = ((torch.from_numpy(loaded_dict["goal_img"]).float() - 127.5)/127.5).to(device).unsqueeze(0)
    next_i = ((torch.from_numpy(loaded_dict["next_img"]).float() - 127.5)/127.5).to(device).unsqueeze(0) 
    image_cat = torch.cat((cur_i, next_i, goal_i), axis=1)   

    #setting robot radius
    robot_size = args.robot_radius*torch.ones(1, 1, 1, 1).to(device) #robot radius for generating velocity commands
    robot_size_te = args.robot_radius*torch.ones(1, 1, 1, 1).to(device) #robot radius for traversability estimation
               
    #setting initial position on local robot coordinate
    px = torch.zeros(1).to(device)
    pz = torch.zeros(1).to(device)
    ry = torch.zeros(1).to(device)

    #calculating our control policy
    with torch.no_grad():
        vwres, ptrav = polinet(image_cat, robot_size, robot_size_te, px, pz, ry)

    vt = vwres.cpu().numpy()[0,0,0,0] #linear velocity to control the robot
    wt = vwres.cpu().numpy()[0,1,0,0] #angular velocity to control the robot
    print("Linear vel.:", vt, "Angular vel.:", wt)

    robot_future_x = []
    robot_future_y = []     
    px = torch.zeros(1).to(device)
    pz = torch.zeros(1).to(device)
    ry = torch.zeros(1).to(device)     
    
    #integrating velocity commands to visualize the generated robot future trajectories by our method
    #Note that the cordinate is on the image corrdinate.  
    for j in range(8):
        ry = ry - vwres[:, 2*j+1, 0, 0] * 0.33
        pz = pz + vwres[:, 2*j, 0, 0] * 0.33 * sinc_apx(-ry/3.1415)
        px = px - vwres[:, 2*j, 0, 0] * 0.33 * sinc_apx(-ry/(2*3.1415)) * torch.sin(-ry / 2.0)

        robot_future_x.append(px.unsqueeze(1))
        robot_future_y.append(pz.unsqueeze(1))

    robot_future_x = torch.cat(robot_future_x, axis=1)                    
    robot_future_y = torch.cat(robot_future_y, axis=1)
    robot_future_xy = torch.cat((robot_future_x, robot_future_y), axis=1)    

    norm = 10.0
    ped_past_xy = torch.clamp(torch.from_numpy(loaded_dict["ped_traj"]).to(device), min=-norm, max=norm)   #pedestrian's past traj.
    robot_past_xy = torch.from_numpy(loaded_dict["robot_traj"]).to(device)                                 #robot's past traj.

    #Estimation of the pedestrian's motion
    #Note that we give normalization for each trajectory
    with torch.no_grad():
        delta_est_ped_future = pednet(ped_past_xy.float().unsqueeze(0)/norm, robot_past_xy.float().unsqueeze(0)/norm, robot_future_xy/norm)        #estimation of the pedestrian's future traj. conditioned on the robot future motion
        delta_est_ped_future_i = pednet(ped_past_xy.float().unsqueeze(0)/norm, robot_past_xy.float().unsqueeze(0)/norm, 0.0*robot_future_xy/norm)  #estimation of the intended pedestrian's future traj.
        
    ped_future_x = torch.cumsum(delta_est_ped_future[:,0:8]/norm, dim=1) + ped_past_xy.float().unsqueeze(0)[:,0:1].repeat(1,8)/norm
    ped_future_y = torch.cumsum(delta_est_ped_future[:,8:16]/norm, dim=1) + ped_past_xy.float().unsqueeze(0)[:,8:9].repeat(1,8)/norm           
    ped_future_xy = torch.clamp(torch.cat((ped_future_x, ped_future_y), axis=1), min=-norm, max=norm)*norm                          #estimated pedestrian's traj.

    ped_future_x_i = torch.cumsum(delta_est_ped_future_i[:,0:8]/norm, dim=1) + ped_past_xy.float().unsqueeze(0)[:,0:1].repeat(1,8)/norm
    ped_future_y_i = torch.cumsum(delta_est_ped_future_i[:,8:16]/norm, dim=1) + ped_past_xy.float().unsqueeze(0)[:,8:9].repeat(1,8)/norm #estimated pedestrian's traj.     
    ped_future_xy_i = torch.clamp(torch.cat((ped_future_x_i, ped_future_y_i), axis=1), min=-norm, max=norm)*norm

    cur_ic = np.concatenate((loaded_dict["cur_img"][0:3], loaded_dict["cur_img"][3:6]), axis=2).transpose(1,2,0)     #current image for visualization
    goal_ic = np.concatenate((loaded_dict["goal_img"][0:3], loaded_dict["goal_img"][3:6]), axis=2).transpose(1,2,0)  #goal image for visualization
    
    #plot on Matplotlib        
    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[0:2, 1])   

    ax1.imshow(cur_ic)
    ax1.title.set_text('current image')
    
    ax2.imshow(goal_ic)
    ax2.title.set_text('subgoal image')
    
    ax4.plot(robot_past_xy[0:8].cpu().numpy(), robot_past_xy[8:16].cpu().numpy(), color='blue', marker="o", label='robot past')
    ax4.plot(robot_future_xy[0,0:8].cpu().numpy(), robot_future_xy[0,8:16].cpu().numpy(), color='cyan', marker="o", label='robot future (ours)')      
    ax4.plot(ped_past_xy[0:8].cpu().numpy(), ped_past_xy[8:16].cpu().numpy(), color='red', marker="o", label='ped. past')
    ax4.plot(ped_future_xy[0,0:8].cpu().numpy(), ped_future_xy[0,8:16].cpu().numpy(), color='orange', marker="o", label='ped. future')    
    ax4.plot(ped_future_xy_i[0,0:8].cpu().numpy(), ped_future_xy_i[0,8:16].cpu().numpy(), color='pink', marker="o", label='ped. future (intended)')           
    ax4.plot(loaded_dict["goal_pose"][0, 3], loaded_dict["goal_pose"][2, 3], color='black', marker="s", label='subgoal pos.')    
    ax4.axis(ymin=-2.5, ymax=4.0)
    ax4.axis(xmin=-2.5,xmax=2.5)
    ax4.legend(loc='upper left')

    plt.show()            
