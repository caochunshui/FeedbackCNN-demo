function [max_en, im_all] = FR_sim(net1, input_data, net2, thd1, thd2, layer_name)
% Run FR simultaneously.
fr_blob_and_layer = layer_name;%Blob name and layer name of selected layer for FR.
blob_name = fr_blob_and_layer;
layer_name = fr_blob_and_layer;

net1.forward(input_data);
layer_index = net1.name2layer_index(layer_name)-1;

% Filter the neurons selected by FSP. This step can be implemented in another way, as follows:
%     blob = net2.blobs(fr_blob_and_layer).get_diff();
%     blob = sum(blob, 3);
%     sum_en = imresize(sum_en, size(blob));
%     blob = sum_en+(blob./max(blob(:)));
%     blob = blob./max(blob(:));
%     blob = mean_thd(blob, thd1);
blob = net2.blobs(fr_blob_and_layer).get_diff();% Get the gradients of neurons selected by FSP.
blob = mean_thd(blob, thd1);% Filtered neurons by a threshold related to mean gradient value of all the neurons selected by FSP.
[sb, index] = sort(blob(:), 'descend');
bigz = find(sb>0);
index = index(bigz);%Obtain the location.

blob_diff = net1.blobs(blob_name).get_diff();     
blob_diff = 0*blob_diff;

blob_diff(index(:)) = 1;% Set all the reserved neurons as target.
net1.blobs(blob_name).set_diff(blob_diff);
net1.set_all_relus_threshold_ratio(thd2);
res = net1.backwardfromto(layer_index, 0);%Perform FR.

net1.reset_all_relu_gates(1);
% Get the visualizaton and energy map.
data_diff = res{1};
im = data_diff;
im = (im-min(im(:)))./(max(im(:))-min(im(:)));
im = im(:, :, [3, 2, 1]);
im = permute(im, [2, 1, 3]);

b = data_diff;
b = permute(b, [2, 1, 3]);
b = sqrt(b(:, :, 1).^2 + b(:, :, 2).^2 + b(:, :, 3).^2);
b = b ./ max(b(:));

im_all = im;
max_en = b;

end