//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include "cudasift/cudasift_export.h"

class CUDASIFT_EXPORT CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

CUDASIFT_EXPORT int iDivUp(int a, int b);
CUDASIFT_EXPORT int iDivDown(int a, int b);
CUDASIFT_EXPORT int iAlignUp(int a, int b);
CUDASIFT_EXPORT int iAlignDown(int a, int b);
CUDASIFT_EXPORT void StartTimer(unsigned int *hTimer);
CUDASIFT_EXPORT double StopTimer(unsigned int hTimer);

#endif // CUDAIMAGE_H
