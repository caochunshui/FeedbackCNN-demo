function [s_iter] = FSP_iter(net2, input_data, target, thd1)

blob_and_layer = 'fc8';
blob_name = blob_and_layer;
layer_name = blob_and_layer;

blob_and_layer2 = 'conv1_2';  
layer_name2 = blob_and_layer2;

net2.reset_all_relu_gates(1);

layer_index = net2.name2layer_index(layer_name)-1;
layer_index2 = net2.name2layer_index(layer_name2)-1;
s_iter = zeros([1, 5]);
for iter = 1:5
    scores = net2.forward(input_data);
    s = scores{1};
    s_iter(iter) = s(target);

    blob_diff = net2.blobs(blob_name).get_diff();
    blob_diff = 0*blob_diff;

    blob_diff(target) = 1;
    net2.blobs(blob_name).set_diff(blob_diff);
    net2.set_all_relus_threshold_ratio(thd1);
    res = net2.backwardfromto(layer_index, layer_index2-1);
end

net2.reset_all_relu_gates(1);
    
end