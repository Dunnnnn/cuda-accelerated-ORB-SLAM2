#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>
#include <functional>
#include <iterator>

#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/types.hpp"

#define W_ORB_SLAM 30
#define BLOCK_WIDTH_BASE 32

#define MAX_KEY_POINTS 10000

typedef struct{
    uint16_t x,y;
    float response;
}keyPoint_t;

typedef struct{
    unsigned int nKeyPoints;
    keyPoint_t pointList[MAX_KEY_POINTS];
}KeyPoints_t;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code == cudaSuccess)
        return;

    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
        exit(code);
}

void* _alloc_host(size_t bytes, int flag = 0);
void* _alloc_device(size_t bytes);

void printGPUinfo();

void detectFAST_gpu(cv::cuda::GpuMat d_img, 
                        KeyPoints_t* p_d_keypoints, 
                        int maxThreshold, int minThreshold, 
                        cv::cuda::GpuMat d_scoreMat);

void detectFAST_gpu_async(cv::cuda::GpuMat d_img, 
                        KeyPoints_t* p_d_keypoints, 
                        int maxThreshold, int minThreshold, 
                        cv::cuda::GpuMat d_scoreMat,
                        cudaStream_t stream);

void computeDescriptors_gpu(cv::cuda::GpuMat d_img, cv::KeyPoint* d_p_kps, int n_kps, cv::cuda::GpuMat d_des);
void computeDescriptors_gpu_async(cv::cuda::GpuMat d_img, cv::KeyPoint* d_p_kps, int n_kps, cv::cuda::GpuMat d_des, cudaStream_t stream);
