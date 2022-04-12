#pragma once
#include "kernel.cuh"

#include <opencv2/core/cuda_stream_accessor.hpp>

class FASTDetector_gpu
{
public:
    cv::cuda::GpuMat d_resized_image;

    cv::cuda::GpuMat d_img;
    cv::cuda::GpuMat d_scoreMat;

    KeyPoints_t *p_h_KeyPoints = nullptr, *p_d_KeyPoints = nullptr;

    cv::cuda::Stream cvStream;
    cudaStream_t cudaStream;

    cv::Ptr<cv::cuda::Filter> filter;

#define MAX_FINAL_KPS_PER_LEVEL 10000
    cv::KeyPoint* d_final_key_points;
    cv::cuda::GpuMat d_des;
    
    FASTDetector_gpu(){
        cvStream = cv::cuda::Stream();
        cudaStream = cv::cuda::StreamAccessor::getStream(cvStream);
        filter = cv::cuda::createGaussianFilter(0, 0, cv::Size(7,7), 2, 2, cv::BorderTypes::BORDER_REFLECT_101);
        d_final_key_points = (cv::KeyPoint*)_alloc_device(sizeof(cv::KeyPoint) * MAX_FINAL_KPS_PER_LEVEL);
        d_des = cv::cuda::GpuMat(MAX_FINAL_KPS_PER_LEVEL, 32, 0);
    }

    ~FASTDetector_gpu(){
        if (p_d_KeyPoints != nullptr){
            cudaFree(p_d_KeyPoints);
            cudaFreeHost(p_h_KeyPoints);
        }
    }

    void uploadSync(cv::Mat h_img){
        if(p_h_KeyPoints == nullptr){
            d_img = cv::cuda::GpuMat(h_img.size(), h_img.type());
            d_scoreMat = cv::cuda::GpuMat(h_img.size(), CV_32SC1);

            p_h_KeyPoints = (KeyPoints_t*)_alloc_host(sizeof(KeyPoints_t), cudaHostAllocDefault);
            p_d_KeyPoints = (KeyPoints_t*)_alloc_device(sizeof(KeyPoints_t));
        }

        d_img.upload(h_img);
        d_scoreMat.setTo(cv::Scalar::all(0));
        gpuErrchk(cudaMemset((void*)p_d_KeyPoints, 0, sizeof(int)));
    }

    KeyPoints_t* detectSync(int maxThreshold, int minThreshold){
        detectFAST_gpu(d_img, p_d_KeyPoints, maxThreshold, minThreshold, d_scoreMat);

        gpuErrchk(cudaMemcpy(p_h_KeyPoints, p_d_KeyPoints, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (p_h_KeyPoints->nKeyPoints > MAX_KEY_POINTS) p_h_KeyPoints->nKeyPoints = MAX_KEY_POINTS;
        gpuErrchk(cudaMemcpy(&p_h_KeyPoints->pointList, &p_d_KeyPoints->pointList, sizeof(keyPoint_t) * p_h_KeyPoints->nKeyPoints, cudaMemcpyDeviceToHost));

        return p_h_KeyPoints;
    }

    void uploadAsync(cv::Mat h_img){
        if(p_h_KeyPoints == nullptr){
            d_img = cv::cuda::GpuMat(h_img.size(), h_img.type());
            d_scoreMat = cv::cuda::GpuMat(h_img.size(), CV_32SC1);

            p_h_KeyPoints = (KeyPoints_t*)_alloc_host(sizeof(KeyPoints_t), cudaHostAllocDefault);
            p_d_KeyPoints = (KeyPoints_t*)_alloc_device(sizeof(KeyPoints_t));
        }

        d_img.upload(h_img, cvStream);
        d_scoreMat.setTo(cv::Scalar::all(0), cvStream);
        gpuErrchk(cudaMemsetAsync((void*)p_d_KeyPoints, 0, sizeof(int), cudaStream));
    }

    void detectAsync(int maxThreshold, int minThreshold){
        detectFAST_gpu_async(d_img, p_d_KeyPoints, maxThreshold, minThreshold, d_scoreMat, cudaStream);

        gpuErrchk(cudaMemcpyAsync(p_h_KeyPoints, p_d_KeyPoints, sizeof(KeyPoints_t), cudaMemcpyDeviceToHost, cudaStream));
    }

    void wait4complete(){
        gpuErrchk(cudaStreamSynchronize(cudaStream));
    }

    KeyPoints_t* getKeyPoints(){
        return p_h_KeyPoints;
    }

    void resizeSync(cv::cuda::GpuMat d_resize_src, cv::Size sz){
        if (d_resized_image.size() != sz){
            d_resized_image = cv::cuda::GpuMat(sz, d_resize_src.type());
        }

        cv::cuda::resize(d_resize_src, d_resized_image, sz, 0, 0, cv::InterpolationFlags::INTER_LINEAR);
    }

    void downloadResizedImg(cv::Mat dst){
        d_resized_image.download(dst);
    }

    void resizeAsync(cv::cuda::GpuMat d_resize_src, cv::Size sz){
        if (d_resized_image.size() != sz){
            d_resized_image = cv::cuda::GpuMat(sz, d_resize_src.type());
            
        }

        cv::cuda::resize(d_resize_src, d_resized_image, sz, 0, 0, cv::InterpolationFlags::INTER_LINEAR, cvStream);
    }

    void downloadResizedImgAsync(cv::Mat dst){
        d_resized_image.download(dst, cvStream);
    }

    void setAsync(cv::Mat h_img, int y1, int y2, int x1, int x2){
        if(p_h_KeyPoints == nullptr){
            // d_img = cv::cuda::GpuMat(h_img.size(), _d_img.type());
            d_scoreMat = cv::cuda::GpuMat(h_img.rowRange(y1, y2).colRange(x1, x2).size(), CV_32SC1);

            p_h_KeyPoints = (KeyPoints_t*)_alloc_host(sizeof(KeyPoints_t), cudaHostAllocDefault);
            p_d_KeyPoints = (KeyPoints_t*)_alloc_device(sizeof(KeyPoints_t));
        }

        d_img = d_resized_image.rowRange(y1, y2).colRange(x1, x2);
        d_scoreMat.setTo(cv::Scalar::all(0), cvStream);
        gpuErrchk(cudaMemsetAsync((void*)p_d_KeyPoints, 0, sizeof(int), cudaStream));
    }

    void computeDescriptorsSync(cv::KeyPoint* p_h_final_pks, int n_final_kps, cv::Mat h_des){
        cv::cuda::GpuMat d_workingMat = d_resized_image;

        gpuErrchk(cudaMemcpy(d_final_key_points, p_h_final_pks, sizeof(cv::KeyPoint) * n_final_kps, cudaMemcpyHostToDevice));
        d_des.rowRange(0, n_final_kps).setTo(cv::Scalar::all(0));

        computeDescriptors_gpu(d_resized_image, d_final_key_points, n_final_kps, d_des);

        d_des.rowRange(0, n_final_kps).download(h_des);
    }

    void uploadKeyPointAsync(cv::KeyPoint* p_h_final_pks, int n_final_kps){
        cv::cuda::GpuMat d_workingMat = d_resized_image;

        gpuErrchk(cudaMemcpyAsync(d_final_key_points, p_h_final_pks, sizeof(cv::KeyPoint) * n_final_kps, cudaMemcpyHostToDevice, cudaStream));
        d_des.rowRange(0, n_final_kps).setTo(cv::Scalar::all(0), cvStream);
    }

    void computeDescriptorsAsync(int n_final_kps, cv::Mat h_des){
        computeDescriptors_gpu_async(d_resized_image, d_final_key_points, n_final_kps, d_des, cudaStream);

        d_des.rowRange(0, n_final_kps).download(h_des, cvStream);
    }

};