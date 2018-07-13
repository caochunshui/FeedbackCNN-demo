#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      Blob<Dtype>*gates=new Blob<Dtype>();
      gates->Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
      caffe_set(gates->count(),Dtype(1.0),gates->mutable_cpu_data());
     
      Blob<Dtype>*buffer=new Blob<Dtype>();
      buffer->Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
      caffe_set(buffer->count(),Dtype(1.0),buffer->mutable_cpu_data());

      Blob<Dtype>*activation=new Blob<Dtype>();
      activation->Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
      caffe_set(activation->count(),Dtype(1.0),activation->mutable_cpu_data());
      this->blobs_.resize(3);
      this->blobs_[0].reset(gates);
      this->blobs_[1].reset(activation);
      this->blobs_[2].reset(buffer);
      activation_on=false;
      threshold_ratio=1;
      bp_mode=0;
}

template <typename Dtype>
void ReLULayer<Dtype>::reset_all_gates(Dtype value){

Dtype* gates_data = this->blobs_[0]->mutable_cpu_data();
Dtype* gates_diff = this->blobs_[0]->mutable_cpu_diff();

const int count =this->blobs_[0]->count();

caffe_set(count,value,gates_data);
caffe_set(count,value,gates_diff);

}


template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* gates_data = this->blobs_[0]->cpu_data();
  Dtype* buffer_data = this->blobs_[2]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
    top_data[i]=gates_data[i]*top_data[i];
    caffe_copy(count,top_data,buffer_data);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* gates_data = this->blobs_[0]->cpu_data();
    Dtype* gates_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
      bottom_diff[i]=gates_data[i]*bottom_diff[i];
    }
    //caffe_copy(count,top_diff,gates_diff);
    caffe_copy(count,bottom_diff,gates_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
