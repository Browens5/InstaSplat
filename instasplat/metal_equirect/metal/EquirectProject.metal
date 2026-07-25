// Equirectangular projection + UT helpers for Apple Silicon.
// Compile on macOS:
//   xcrun -sdk macosx metal -c EquirectProject.metal -o EquirectProject.air
//   xcrun -sdk macosx metallib EquirectProject.air -o EquirectProject.metallib
//
// The Python trainer calls the torch UT path today; this library is the
// acceleration target for means_cam → mean_2d / cov_2d on the GPU.

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

inline float2 equirect_project(float3 dir, float width, float height) {
    float3 d = normalize(dir);
    float lon = atan2(d.x, d.z);
    float lat = asin(clamp(-d.y, -1.0f, 1.0f));
    float u = (lon / (2.0f * PI) + 0.5f) * width;
    float v = (0.5f - lat / PI) * height;
    return float2(u, v);
}

struct GaussianCam {
    float3 mean;
    float3x3 cov;
};

kernel void project_means_equirect(
    device const float3 *means_cam [[buffer(0)]],
    device float2 *uv_out [[buffer(1)]],
    constant float &width [[buffer(2)]],
    constant float &height [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uv_out[id] = equirect_project(means_cam[id], width, height);
}

// 7 sigma points per Gaussian (mean + ±chol columns), then project.
kernel void unscented_project_equirect(
    device const float3 *means_cam [[buffer(0)]],
    device const float3x3 *chol_scaled [[buffer(1)]],  // chol( (3+λ) Σ )
    device float2 *sigma_uv [[buffer(2)]],             // N * 7
    constant float &width [[buffer(3)]],
    constant float &height [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float3 m = means_cam[id];
    float3x3 L = chol_scaled[id];
    float3 offsets[7];
    offsets[0] = float3(0.0);
    offsets[1] = L[0];
    offsets[2] = L[1];
    offsets[3] = L[2];
    offsets[4] = -L[0];
    offsets[5] = -L[1];
    offsets[6] = -L[2];
    for (uint k = 0; k < 7; ++k) {
        sigma_uv[id * 7 + k] = equirect_project(m + offsets[k], width, height);
    }
}
