function seg= saliency_cut(img,sal,output)

      seg=caffe_('saliency_cut',img,sal,output);



end
