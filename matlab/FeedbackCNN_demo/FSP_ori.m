function [res] = FSP_ori(net2, input_data, target)
% This is the original implementation of FSP.
blob_and_layer = 'fc8';
blob_name = blob_and_layer;
layer_name = blob_and_layer;

blob_and_layer2 = 'conv1_2';  
blob_name2 = blob_and_layer2;  
layer_name2 = blob_and_layer2;

layer_index = net2.name2layer_index(layer_name)-1;
layer_index2 = net2.name2layer_index(layer_name2)-1;

for iter = 1:5
    scores2 = net2.forward(input_data);
    blob_diff = net2.blobs(blob_name).get_diff();
    blob_diff = 0*blob_diff;
    blob_diff(target) = 1;
    net2.blobs(blob_name).set_diff(blob_diff);
    net2.set_all_relus_threshold_ratio(0);
    res = net2.backwardfromto(layer_index, layer_index2-1);
end

scores2 = net2.forward(input_data);

fc8_diff = zeros([1000, 1]);
fc8_diff(target) = 1;
net2.set_all_relus_bp_mode(0);
res = net2.backward({fc8_diff});
net2.set_all_relus_bp_mode(2);

end