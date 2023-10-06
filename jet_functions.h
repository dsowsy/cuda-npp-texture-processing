// jet_functions.h

#ifndef JET_FUNCTIONS_H
#define JET_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

void apply_jet_colormap_host(const float* gray_image, float* rgb_image, int width, int height);

#ifdef __cplusplus
}
#endif

__device__ void jet_colormap(float gray_value, float &r, float &g, float &b);
__global__ void apply_jet_colormap(const float* gray_image, float* rgb_image, int width, int height);

#endif // JET_FUNCTIONS_H
