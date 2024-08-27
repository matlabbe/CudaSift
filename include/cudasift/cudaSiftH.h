#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudasift/cudautils.h"
#include "cudasift/cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

CUDASIFT_EXPORT int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float edgeLimit, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
CUDASIFT_EXPORT void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float edgeLimit, float lowestScale, float subsampling, float *memoryTmp);
CUDASIFT_EXPORT double ScaleDown(CudaImage &res, CudaImage &src, float variance);
CUDASIFT_EXPORT double ScaleUp(CudaImage &res, CudaImage &src);
CUDASIFT_EXPORT double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);
CUDASIFT_EXPORT double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
CUDASIFT_EXPORT double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
CUDASIFT_EXPORT double RescalePositions(SiftData &siftData, float scale);
CUDASIFT_EXPORT double LowPass(CudaImage &res, CudaImage &src, float scale);
CUDASIFT_EXPORT void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
CUDASIFT_EXPORT double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);
CUDASIFT_EXPORT double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

#endif
