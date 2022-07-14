import os
import glob
import time
from datetime import datetime
from PIL import Image
import csv
import sys

sys.path.append(".") 
import torch
import os
import time
import os.path as osp
import numpy as np
import torch.nn.functional as F
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env


import torchrl.policies as policies
import torchrl.networks as networks
import gym
from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.




args = get_args()
params = get_params(args.config)
env=gym.make(params['env_name'])
task_list=["dir_15_mixed","dir_45_mixed","dir_75_mixed","dir_105_mixed","dir_135_mixed","dir_165_mixed","dir_195_mixed","dir_225_mixed","dir_255_mixed", "dir_285_mixed","dir_315_mixed","dir_345_mixed"]
task_num=len(task_list)
representation_shape= params['representation_shape']
embedding_shape=params['embedding_shape']
params['p_state_net']['base_type']=networks.MLPBase
params['task_net']['base_type']=networks.MLPBase
params['p_action_net']['base_type']=networks.MLPBase
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
pf_state = networks.Net(
	input_shape=env.observation_space.shape[0], 
	output_shape=representation_shape,
	**params['p_state_net']
)

pf_action=policies.ActionRepresentationGuassianContPolicy(
	input_shape = representation_shape + embedding_shape,
	output_shape = 2 * env.action_space.shape[0],
	**params['p_action_net'] 
)
experiment_id = str(args.id)
experiment_id_v2 = experiment_id + "_mixed"
model_dir="log/"+experiment_id+"/"+params['env_name']+"/"+str(args.seed)+"/model/"

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_finish.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_finish.pth", map_location='cpu'))

pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_8060.pth", map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_8060.pth", map_location='cpu'))

# pf_state.load_state_dict(torch.load(model_dir + "model_pf_state_best.pth", map_location='cpu'))
# pf_action.load_state_dict(torch.load(model_dir + "model_pf_action_best.pth", map_location='cpu'))

############################# save images for gif ##############################


def save_gif_images(env_name, max_ep_len):

	print("============================================================================================")
	device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
	
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.cuda:
		torch.backends.cudnn.deterministic=True

	# make directory for saving gif images
	gif_images_dir = "gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir = gif_images_dir + '/' + experiment_id_v2 + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir_list=[]
	
	for i in range(len(task_list)):
		# gif_images_dir_list[i]=gif_images_dir+"/"+cls_list[i]+"/"
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_images_dir_list[i]):
			os.makedirs(gif_images_dir_list[i])

	# make directory for gif
	gif_dir = "gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id_v2 + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir_list=[]
	
	for i in range(len(task_list)):
        # gif_dir_list[i]=gif_dir+"/"+cls_list[i]+"/"
		gif_dir_list.append(gif_dir+"/"+task_list[i]+"/")
		if not os.path.exists(gif_dir_list[i]):
			os.makedirs(gif_dir_list[i])

	if params["save_embedding"]:
		embed_dir = "embedding"+'/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + env_name  + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)
		embed_dir = embed_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed_dir):
			os.makedirs(embed_dir)	

		embed4q_dir = "embedding4q"+'/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + env_name  + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)
		embed4q_dir = embed4q_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(embed4q_dir):
			os.makedirs(embed4q_dir)	

	if params["save_velocity"]:
		velocity_dir = "velocity"+'/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + experiment_id_v2 + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + env_name  + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)
		velocity_dir = velocity_dir + '/' + str(args.seed) + '/'
		if not os.path.exists(velocity_dir):
			os.makedirs(velocity_dir)	

		average_v_csv_path = velocity_dir+ "/average_velocity.csv"
		average_v_file = open(average_v_csv_path,"a")
		average_v_writer = csv.writer(average_v_file)
		average_v_writer.writerow(["task","v_mean","v_std"])

	pre_embeddings=[]
	pre_embedding=torch.Tensor([4.936531,0.71696174,0.3415116]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([4.9628754,0.054794494,0.60569173]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([2.2239823,0.7987723,-4.406344]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([2.1639457,-0.08721691,-4.506632]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([3.5592418,-2.3015976,-2.6522532]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([4.315548,-1.7585125,-1.8120922]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([4.736697,-1.3598969,-0.84520847]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([4.574261,-1.8774043,-0.742622]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([1.0700808,-3.7418633,-3.1390104]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([4.5968156,1.3600149,-1.4211421]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.030164212,-1.3482533,-4.814697]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	pre_embedding=torch.Tensor([-0.17253584,0.4935394,-4.97259]).unsqueeze(0)
	pre_embeddings.append(pre_embedding)
	embeddings=[]
	for i in range(11):
		embedding = (pre_embeddings[i]+pre_embeddings[i+1])/2
		embedding = 5 * F.normalize(embedding)
		embeddings.append(embedding)
	embedding = (pre_embeddings[11]+pre_embeddings[0])/2
	embedding = 5 * F.normalize(embedding)
	embeddings.append(embedding)
 
	pre_embeddings4q=[]
	pre_embedding4q=torch.Tensor([2.8113074,-0.5523925,-4.0977325]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([2.052821,-1.4819754,-4.311574]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([1.0253621,-0.1841923,-4.8902664]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([2.1639457,-0.08721691,-4.506632]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.21361054,-0.92456406,-4.909129]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([1.2884551,-0.56611884,-4.797853]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([2.0226905,-0.83962744,-4.4948583]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([1.0968341,-1.1697899,-4.735879]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.0088049695,-1.1049261,-4.876378]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([2.2200308,-0.4476421,-4.457699]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([-0.4588258,-0.94878304,-4.887667]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	pre_embedding4q=torch.Tensor([0.93127483,0.752826,-4.8544803]).unsqueeze(0)
	pre_embeddings4q.append(pre_embedding4q)
	embeddings4q=[]
	for i in range(11):
		embedding4q = (pre_embeddings4q[i]+pre_embeddings4q[i+1])/2
		embedding4q = 5 * F.normalize(embedding4q)
		embeddings4q.append(embedding4q)
	embedding4q = (pre_embeddings4q[11]+pre_embeddings4q[0])/2
	embedding4q = 5 * F.normalize(embedding4q)
	embeddings4q.append(embedding4q)

	for i in range(task_num):
		if params["save_embedding"]:
			embed_csv_path = embed_dir + '/' + task_list[i] + ".csv"
			embed_file = open(embed_csv_path, "w")
			embed_writer = csv.writer(embed_file)
			embed4q_csv_path = embed4q_dir + '/' + task_list[i] + ".csv"
			embed4q_file = open(embed4q_csv_path, "w")
			embed4q_writer = csv.writer(embed4q_file)
		if params["save_velocity"]:
			velocity_csv_path = velocity_dir+ '/' + task_list[i] + ".csv"
			velocity_file = open(velocity_csv_path,'w')
			velocity_writer = csv.writer(velocity_file)
		embedding=embeddings[i]
		embedding4q=embeddings4q[i]
		ob=env.reset()
		with torch.no_grad():
			for t in range(1, max_ep_len+1):
				representation = pf_state.forward(torch.Tensor( ob ).to("cpu").unsqueeze(0))
				out=pf_action.explore(representation,embedding)
				act=out["action"]
				act = act.detach().cpu().numpy()
				next_ob, _, done, info = env.step(act)
				if params["save_velocity"]:
					x_velocity = info['x_velocity']
					velocity_writer.writerow([x_velocity])
				# img = env.render(mode = 'rgb_array')
				# img = Image.fromarray(img)
				# img.save(gif_images_dir_list[i] + '/' + experiment_id + '_' + task_list[i] + str(t).zfill(6) + '.jpg')
				ob=next_ob
				if done:
					break
			x = info['x_position']
			y = info['y_position']
			dir = np.arctan(y/x)/ np.pi * 180
			if x<0 and y>0:
				dir+=180
			elif x<0 and y<0:
				dir+=180
			elif x>0 and y<0:
				dir+=360
			
			print("task", i, "direction:", dir)

		if params["save_embedding"]:
			embedding = embedding.squeeze(0)
			embedding = embedding.detach().cpu().numpy()
			embed_writer.writerow(embedding)
			embed_file.close()
			embedding4q = embedding4q.squeeze(0)
			embedding4q = embedding4q.detach().cpu().numpy()
			embed4q_writer.writerow(embedding4q)
			embed4q_file.close()
		if params["save_velocity"]:
			velocity_file.close()
			velocity_file = open(velocity_csv_path,'r')
			velocity_list = np.loadtxt(velocity_file)
			velocity_list = velocity_list[100:]
			average_v_writer.writerow([task_list[i], np.mean(velocity_list), np.std(velocity_list)])


	env.close()











######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	gif_num = args.seed    
	experiment_id=str(args.id)

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = 250
	step = 1
	frame_duration = 60

	# input images
	gif_images_dir = "gif_images/" + experiment_id_v2 + '/' + env_name +"/"
	gif_images_dir_list=[]
	for i in range(len(task_list)):
		gif_images_dir_list.append(gif_images_dir+"/"+task_list[i]+"/*.jpg")

	# output gif path
	gif_dir = "gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id_v2 + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	gif_path_list=[]
	for i in range(len(task_list)):
		gif_path_list.append(gif_dir+"/"+task_list[i]+"/"+experiment_id_v2+'_'+task_list[i]+ '_gif_' + str(gif_num) + '.gif')
	
	img_paths_list=[]
	for i in range(len(task_list)):

		img_paths_list.append(sorted(glob.glob(gif_images_dir_list[i]))) 
		img_paths_list[i] = img_paths_list[i][:total_timesteps]
		img_paths_list[i] = img_paths_list[i][::step]

		img, *imgs = [Image.open(f) for f in img_paths_list[i]]
		img.save(fp=gif_path_list[i], format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)
		print("saved gif at : ", gif_path_list[i])



if __name__ == '__main__':
	env_name = params["env_name"]
	max_ep_len = 1000          
	save_gif_images(env_name,  max_ep_len)
	# save_gif(env_name)


