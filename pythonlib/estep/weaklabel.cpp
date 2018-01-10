#include<float.h>
#include<algorithm> // c++ h file for random_shuffle, nth_element
#include<iostream>
//#include<sys/time.h>

extern "C" {
float *e_step(float *map,const int *shape,const int *stride,const int *label,const bool suppress_others,const int num_iter,const float margin_others,float bg,float fg){ // from the shape and stride of map, we can infer those of label
   // struct timeval s;
   // gettimeofday(&s,NULL);
    int total_pixel_each_image = shape[1] * shape[2];

    // rescael the stride
    int *stride_ = (int *)malloc(4 * sizeof(int));
    for(int t=0; t<4; t++){
        stride_[t] = stride[t] / sizeof(float);
    }

    // change b x h x w label to b x c label
    int *label_ = (int *)malloc(shape[0]*shape[3]*sizeof(int));
    for(int b=0; b<shape[0]; b++){
        for(int c=0; c<shape[3]; c++){
            *(label_+b*shape[3]+c) = 0;
        }
        for(int h=0; h<shape[1]; h+=2){
            for(int w=0; w<shape[2]; w+=2){
                *(label_+b*shape[3]+*(label+b*total_pixel_each_image+h*shape[2]+w)) = 1;
            }
        }
    }


    for(int b=0; b<shape[0]; b++){ // the first element of shape is batch_size
        const int *label_single_sparse = label_ + b*shape[3];
        int *label_single = (int *)malloc(shape[3]*sizeof(int));
        int present_label_num = 0;
        for(int k=0; k<shape[3]; k++){
            if(*(label_single_sparse+k) > 0){
                *(label_single+present_label_num) = k;
                present_label_num ++;
            }
        }
        float *min_of_present_label_ = (float *)malloc(total_pixel_each_image * sizeof(float));
        if(suppress_others){ // suppress the absent label_
            for(int i=0; i<total_pixel_each_image; i++){
                *(min_of_present_label_+i) = FLT_MAX;
                for(int c=0; c<shape[3]; c++){ // the last element of shape is channles
                    if(label_single_sparse[c] > 0){ // the label_ is in the image
                        float tmp = *(map + b*stride_[0] + i*stride_[2] +c);
                        if( tmp < *(min_of_present_label_+i)){
                            *(min_of_present_label_+i) = tmp;
                        }
                    }
                }
                for(int c=0; c<shape[3]; c++){
                    if(label_single_sparse[c] < 1){
                        if(*(map+b*stride_[0]+i*stride_[2]+c) > *(min_of_present_label_+i)){
                            *(map+b*stride_[0]+i*stride_[2]+c) = *(min_of_present_label_+i) - margin_others;
                        }
                    }
                }
                  
            }
        }

        float *extrenum = min_of_present_label_; // reuse the memeroy
        float *diff = (float *)malloc(total_pixel_each_image*sizeof(float));
        float mean_max_0 = 0;
        for(int i=0; i<total_pixel_each_image; i++){
            *(extrenum+i) = FLT_MIN;
            for(int c=0; c<shape[3]; c++){
                float tmp = *(map + b*stride_[0]+i*stride_[2] +c);
                if(tmp > *(extrenum+i)){
                    *(extrenum+i) = tmp;
                }
            }
            mean_max_0 += *(extrenum+i);
        }
        mean_max_0 /= total_pixel_each_image;

        for(int j=0; j<num_iter; j++){
            std::random_shuffle(label_single+1,label_single+present_label_num); // random_shuffle except for the bg
            for(int c=0; c<present_label_num; c++){ // generate feature map from weak label
                for(int i=0; i<total_pixel_each_image; i++){
                    *(diff+i) = *(extrenum+i) - *(map+b*stride_[0]+i*stride_[2]+*(label_single+c)); 
                }
                const int nth = (*(label_single+c) == 0) ? bg * total_pixel_each_image : fg * total_pixel_each_image;
                std::nth_element(diff,diff+nth,diff+total_pixel_each_image);
                for(int i=0; i<total_pixel_each_image; i++){
                    *(map+b*stride_[0]+i*stride_[2]+*(label_single+c)) += *(diff+nth);
                }

                for(int i=0; i<total_pixel_each_image; i++){
                    float tmp = *(map+b*stride_[0]+i*stride_[2]+*(label_single+c));
                    if(tmp > *(extrenum+i)){
                        *(extrenum+i) = tmp;
                    }
                }
            }
        }
        double mean_max_1 = 0;
        for(int i=0; i<total_pixel_each_image; i++){
            mean_max_1 += *(extrenum+i);
        }
        mean_max_1 /= total_pixel_each_image;

        for(int i=0; i<total_pixel_each_image; i++){
            for(int c=0; c<shape[3]; c++){
                *(map+b*stride_[0]+i*stride_[2]+c) += (mean_max_0 - mean_max_1);
            }
        }

        free(label_single);
        free(min_of_present_label_);
        free(diff);
    }
//    struct timeval e;
//    gettimeofday(&e,NULL);
//    std::cout<<"dd time:"<<e.tv_usec - s.tv_usec<<std::endl;
    return map;
}
}
