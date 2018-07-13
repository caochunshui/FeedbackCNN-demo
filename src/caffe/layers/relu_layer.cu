#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {






template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope,const Dtype* gates,Dtype* activation_data,bool activation_on) {
  CUDA_KERNEL_LOOP(index, n) {
    if(activation_on){
    out[index] = activation_data[index]*gates[index]*(in[index] > 0 ? in[index] : in[index] * negative_slope);
    }else{
   
    out[index] = gates[index]*(in[index] > 0 ? in[index] : in[index] * negative_slope);
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* gates_data = this->blobs_[0]->gpu_data();
  Dtype* buffer_data = this->blobs_[2]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  Dtype* activation_data =this->blobs_[1]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope,gates_data,activation_data,activation_on);
  CUDA_POST_KERNEL_CHECK;
    caffe_copy(count,top_data,buffer_data);
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope,const Dtype* gates, Dtype* activation_data ,bool activation_on) {
   if(activation_on){
       CUDA_KERNEL_LOOP(index, n) {
           activation_data[index]=in_diff[index]>0?Dtype(1.0):Dtype(0.0);
           out_diff[index] = activation_data[index]*gates[index]*(in_diff[index] * ((in_data[index]>0)
           + (in_data[index] <= 0) * negative_slope));}
        

   }else{
        CUDA_KERNEL_LOOP(index, n) {
           out_diff[index] = gates[index]*(in_diff[index] * ((in_data[index] > 0)
           + (in_data[index] <= 0) * negative_slope));}
        }  
}


template <typename Dtype>
__global__ void Threshold1(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? in[index] : 0;
  }
}

template <typename Dtype>
__global__ void Threshold2(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

template <typename Dtype>
__global__ void Threshold3(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] >= threshold ? 1 : 0;
  }
}

template <typename Dtype>
__global__ void drop_edge_pixels(const int count, const int n,const int c, const int h,const int w,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
   
   int num=index/(c*h*w);
   int ch=(index-num*c*h*w)/(h*w);
   int hei=(index-num*c*h*w-ch*h*w)/h;
   int wid=index-num*c*h*w-ch*h*w-hei*w;
   if (hei==0||hei==h-1||wid==0||wid==w-1){
   
    out[index] =0;
   }else{
    out[index]=in[index];
   }

  }
}





template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    const Dtype* gates_data = this->blobs_[0]->gpu_data();
    Dtype* mutable_gates_data = this->blobs_[0]->mutable_gpu_data();
    Dtype* gates_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    Dtype diff_nonezeros=0;
    Dtype diff_mean=0;
    Dtype threshold=0;   
   // Dtype* activation_data = activation.mutable_gpu_data();
   Dtype* activation_data =this->blobs_[1]->mutable_gpu_data();
    if(this->bp_mode==0){
      ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_data, bottom_diff, negative_slope,gates_data,activation_data,activation_on);
      CUDA_POST_KERNEL_CHECK;




    }
    if(this->bp_mode==1){

       if (this->blobs_[0]->height()>1&&this->blobs_[0]->width()>1){
           // drop edge pixels
         
           drop_edge_pixels<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,this->blobs_[0]->num(),this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width(),top_diff, top_diff);
           CUDA_POST_KERNEL_CHECK; 
       }




    //nonezeros' mean
         Threshold2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, Dtype(0), top_diff, gates_diff);
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_asum(count,gates_diff,&diff_nonezeros);

    Threshold1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, Dtype(0), top_diff, gates_diff);
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_asum(count,gates_diff,&diff_mean);
    
    diff_mean=diff_mean/diff_nonezeros;
    //threshold
    threshold=this->threshold_ratio*diff_mean;

   
    Threshold3<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold, gates_diff, mutable_gates_data);

    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_data, bottom_diff, negative_slope,gates_data,activation_data,activation_on);
      CUDA_POST_KERNEL_CHECK;
    
   }
    if(this->bp_mode==2){
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope,gates_data,activation_data,activation_on);
    CUDA_POST_KERNEL_CHECK;

 
       if (this->blobs_[0]->height()>1&&this->blobs_[0]->width()>1){
           // drop edge pixels
         
           drop_edge_pixels<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,this->blobs_[0]->num(),this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width(),bottom_diff, activation_data);
           CUDA_POST_KERNEL_CHECK; 
       

       }else{
       caffe_copy(count,bottom_diff,activation_data);
       }




   
    Threshold2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, Dtype(0), activation_data, gates_diff);
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_asum(count,gates_diff,&diff_nonezeros);

    Threshold1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, Dtype(0), activation_data, gates_diff);
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_asum(count,gates_diff,&diff_mean);
    
    diff_mean=diff_mean/diff_nonezeros;
    threshold=this->threshold_ratio*diff_mean;

   
    Threshold3<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold, gates_diff, mutable_gates_data);
   }
 //    caffe_copy(count,top_diff,gates_diff);
//    caffe_copy(count,bottom_diff,gates_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
