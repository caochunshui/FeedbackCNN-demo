function [res, se] = FSP_SumEnergy(net2, input_data, target, thd1, thd2)
% This is an updated implementation of FSP. Perform FSP with two stages.

blob_and_layer = 'fc8';
blob_name = blob_and_layer;
layer_name = blob_and_layer;

blob_and_layer2 = 'conv1_2';  %The second stage is performing FSP from 'fc8' to 'conv1_1'.
layer_name2 = blob_and_layer2;

blob_and_layer3 = 'conv4_3';  %The first stage is performing FSP from 'fc8' to 'conv3_3'.
layer_name3 = blob_and_layer3;

layer_index = net2.name2layer_index(layer_name)-1;
layer_index2 = net2.name2layer_index(layer_name2)-1;
layer_index3 = net2.name2layer_index(layer_name3)-1;

%The first stage.
for iter = 1:1
    scores2 = net2.forward(input_data);
    blob_diff = net2.blobs(blob_name).get_diff();
    blob_diff = 0*blob_diff;

    blob_diff(target) = 1;% Set target.
    net2.blobs(blob_name).set_diff(blob_diff);
    net2.set_all_relus_threshold_ratio(thd1);
    res = net2.backwardfromto(layer_index, layer_index3-1);%Perform FSP.
end

%The second stage.
for iter = 1:1
    scores2 = net2.forward(input_data);
    blob_diff = net2.blobs(blob_name).get_diff();
    blob_diff = 0*blob_diff;
    blob_diff(target) = 1;% Set target.
    net2.blobs(blob_name).set_diff(blob_diff);
    net2.set_all_relus_threshold_ratio(thd2);
    res = net2.backwardfromto(layer_index, layer_index2-1);%Perform FSP.
end

scores2 = net2.forward(input_data);
fc8_diff = zeros([1000, 1]);
fc8_diff(target) = 1;
net2.set_all_relus_bp_mode(0);
res = net2.backward({fc8_diff});
net2.set_all_relus_bp_mode(2);
  
% Get the Summation Energy Map.
blob_target = {'conv4_2', 'conv4_1', 'conv3_3', 'conv3_2', 'conv3_1', 'conv2_2', 'conv2_1'};
all_en = zeros(224, 224);
for i = 1:7
    blob_diff = net2.blobs(blob_target{i}).get_data();
    b = sum(blob_diff, 3);
    b = b./max(b(:));
    b = imresize(b', [224, 224], 'bilinear');
    all_en = all_en+b;
end
se = all_en./max(all_en(:));
      
end