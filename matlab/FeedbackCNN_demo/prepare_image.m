function [ crops_data] = prepare_image( im )
%PREPARE_IMAGE Summary of this function goes here
%   Detailed explanation goes here
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
% d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');

IMAGE_DIM = 256;
CROPPED_DIM = 224;
if size(im,3)<3
    tmp=zeros([size(im,1),size(im,2),3]);
    tmp(:,:,1)=im;
    tmp(:,:,2)=im;
    tmp(:,:,3)=im;
    im=tmp;
end

im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data(:,:,1) = im_data(:,:,1) - 104;  % subtract mean_data (already in W x H x C, BGR)
im_data(:,:,2) = im_data(:,:,2) - 117;
im_data(:,:,3) = im_data(:,:,3) - 123;
crops_data(:,:,:,1)= imresize(im_data, [CROPPED_DIM CROPPED_DIM], 'bilinear');

% oversample (4 corners, center, and their x-axis flips)
end









