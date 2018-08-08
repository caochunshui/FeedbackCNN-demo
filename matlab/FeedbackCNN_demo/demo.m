addpath('..');

use_gpu=1;
if use_gpu
    caffe.set_mode_gpu();
    gpu_id = 3;  
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

caffe.reset_all();
close all;

%Pre-trained model.
model_dir = '../../models/vgg/';
net_model1 = [model_dir 'deploy.prototxt'];
net_model2 = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'vgg_imagenet.caffemodel'];

phase = 'test'; 
if ~exist(net_weights, 'file')
    error('Please download CaffeNet from Model Zoo before you run this demo');
end

net1 = caffe.Net(net_model1, net_weights, phase);%Model for FR.
net1.set_all_relus_bp_mode(1);% Set the operation mode of all relu layers as the FR mode.
net2 = caffe.Net(net_model2, net_weights, phase);%Model for FSP.
net2.set_all_relus_bp_mode(2);% Set the operation mode of all relu layers as the FSP mode.

im_path = './demo_image/997_265_ori.jpg';   
target = 265;% Set the label of target object.
image_name = 997; % Use a number as the name of this image.

im_ori = imread(im_path);
input_data = {prepare_image(im_ori)};

fsp_thd1 = 1;% The threshold of the first stage of updated FSP.
fsp_thd2 = 1;% The threshold of the second stage of updated FSP.
FR_separately_thd1 = 1;% The threshold for selecting neurons after running FSP. 
FR_separately_thd2 = 1;% The threshold for runnning FR separately.
FR_simultaneously_thd1 = 0.1;% The threshold for selecting neurons after running FSP. 
FR_simultaneously_thd2 = 0.1;% The threshold for runnning FR simultaneously.
%These parameters can be determined by estimating the size of objects using summation_energy maps.

[res, summation_energy] = FSP_SumEnergy(net2, input_data, target, fsp_thd1, fsp_thd2);%Perform FSP and generate Summation Energy Map.
[max_en, im_all] = FR_seq(net1, input_data, net2, FR_separately_thd1, FR_separately_thd2, 'conv5_1'); %Perform FR on neurons selected by FSP separately and generate the merged FSP-FR energy and visualization map.

imwrite(max_en, sprintf('./results/%d_%d_FR_separately_energy.jpg', image_name, target), 'jpg');
imwrite(im_all, sprintf('./results/%d_%d_FR_separately_show.jpg', image_name, target), 'jpg');
[max_en, im_all] = FR_sim(net1, input_data, net2, FR_simultaneously_thd1, FR_simultaneously_thd2, 'conv5_1');%Perform FR on neurons selected by FSP simultaneously and generate the FSP-FR energy and visualization map.

imwrite(max_en, sprintf('./results/%d_%d_FR_simultaneously_energy.jpg', image_name, target), 'jpg');
imwrite(im_all, sprintf('./results/%d_%d_FR_simultaneously_show.jpg', image_name, target), 'jpg');

imwrite(im_ori, sprintf('./results/%d_%d_ori.jpg', image_name, target), 'jpg');
imwrite(summation_energy, sprintf('./results/%d_%d_summation_energy.jpg', image_name, target), 'jpg');  
net2.reset_all_relu_gates(1); %Reset the status of all the feedback gates.

caffe.reset_all();