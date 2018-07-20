addpath('..');

use_gpu=1;
if use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

caffe.reset_all();
close all;

model_dir = '../../models/vgg/';
net_model1 = [model_dir 'deploy.prototxt'];
net_model2 = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'vgg_imagenet.caffemodel'];

phase = 'test'; 
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

net2 = caffe.Net(net_model2, net_weights, phase);
net2.set_all_relus_bp_mode(2);

im_path='./demo_image/997_265_ori.jpg';    
target=265;

im=imread(im_path);
input_data = {prepare_image(im)};
 
iteration_data = FSP_iter(net2,input_data,target,0.000000000001);

net2.reset_all_relu_gates(1);

caffe.reset_all();