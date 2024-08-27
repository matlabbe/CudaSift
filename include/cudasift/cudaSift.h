#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudasift/cudaImage.h"

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float gaussian_diff;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[2];
  float data[128];
} SiftPoint;

class SiftData{
public:
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data = NULL;  // Managed data
#else
  SiftPoint *h_data = NULL;  // Host (CPU) data
  SiftPoint *d_data = NULL;  // Device (GPU) data
#endif
};

CUDASIFT_EXPORT void InitCuda(int devNum = 0);
CUDASIFT_EXPORT float * AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
CUDASIFT_EXPORT void FreeSiftTempMemory(float *memoryTmp);
CUDASIFT_EXPORT void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float edgeLimit = 10.0f, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0);
CUDASIFT_EXPORT void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
CUDASIFT_EXPORT void FreeSiftData(SiftData &data);
CUDASIFT_EXPORT void PrintSiftData(SiftData &data);
CUDASIFT_EXPORT double MatchSiftData(SiftData &data1, SiftData &data2);
CUDASIFT_EXPORT double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
