// Equirectangular projection + fused soft-OIT splat for Apple Silicon.
// Compile on macOS:
//   xcrun -sdk macosx metal -c EquirectProject.metal -o EquirectProject.air
//   xcrun -sdk macosx metallib EquirectProject.air -o EquirectProject.metallib
//
// Kernels:
//   project_means_equirect / unscented_project_equirect — means_cam → UV
//   soft_oit_accumulate — footprint soft splat into color/weight buffers
//   soft_oit_normalize — color / (weight + eps)

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

struct SoftOITUniforms {
    uint n;
    uint width;
    uint height;
    int footprint;  // half-extent in pixels (clamped)
    float depth_tau;
};

// One thread per Gaussian: soft-splat a square footprint into atomic accumulators.
// color_acc: Npix * 3 (RGB), weight_acc: Npix
kernel void soft_oit_accumulate(
    device const float2 *mean_2d [[buffer(0)]],
    device const float4 *cov2d_pack [[buffer(1)]],  // (a, b, c, unused) row-major 2x2
    device const float *radius [[buffer(2)]],
    device const float *opacity [[buffer(3)]],
    device const float3 *color [[buffer(4)]],
    device const float *depth [[buffer(5)]],
    device const uchar *valid [[buffer(6)]],
    device atomic_float *color_acc [[buffer(7)]],
    device atomic_float *weight_acc [[buffer(8)]],
    constant SoftOITUniforms &U [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= U.n) return;
    if (valid[id] == 0) return;

    float2 mu = mean_2d[id];
    if (!isfinite(mu.x) || !isfinite(mu.y)) return;

    float r = clamp(radius[id], 1.0f, float(U.footprint));
    if (!(r > 0.5f)) return;

    float4 cp = cov2d_pack[id];
    // Symmetrize + eps toward SPD (parity with torch / CPU reference)
    float a = cp.x + 1e-4f;
    float b01 = cp.y;
    float c = cp.z + 1e-4f;
    float det = a * c - b01 * b01;
    if (det < 1e-12f) return;
    float inv00 = c / det;
    float inv11 = a / det;
    float inv01 = -b01 / det;

    float op = opacity[id] * exp(-U.depth_tau * max(depth[id], 0.01f));
    float3 col = color[id];
    int half = int(ceil(r));
    half = min(half, U.footprint);
    int W = int(U.width);
    int H = int(U.height);

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            float px = mu.x + float(dx);
            float py = mu.y + float(dy);
            float du = px - mu.x;
            // wrap longitude
            du = du - float(W) * round(du / float(W));
            float dv = py - mu.y;
            if (du * du + dv * dv > r * r) continue;
            float py_r = round(py);
            if (py_r < 0.0f || py_r >= float(H)) continue;
            float maha = inv00 * du * du + 2.0f * inv01 * du * dv + inv11 * dv * dv;
            float alpha = clamp(op * exp(-0.5f * maha), 0.0f, 0.99f);
            if (alpha < 1e-5f) continue;
            int ix = int(round(px));
            ix = ((ix % W) + W) % W;
            int iy = clamp(int(py_r), 0, H - 1);
            uint pix = uint(iy * W + ix);
            atomic_fetch_add_explicit(color_acc + pix * 3 + 0, col.x * alpha, memory_order_relaxed);
            atomic_fetch_add_explicit(color_acc + pix * 3 + 1, col.y * alpha, memory_order_relaxed);
            atomic_fetch_add_explicit(color_acc + pix * 3 + 2, col.z * alpha, memory_order_relaxed);
            atomic_fetch_add_explicit(weight_acc + pix, alpha, memory_order_relaxed);
        }
    }
}

kernel void soft_oit_normalize(
    device const float *color_acc [[buffer(0)]],
    device const float *weight_acc [[buffer(1)]],
    device float3 *rgb_out [[buffer(2)]],
    constant uint &n_pix [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_pix) return;
    float w = weight_acc[id] + 1e-6f;
    float3 c = float3(color_acc[id * 3 + 0], color_acc[id * 3 + 1], color_acc[id * 3 + 2]) / w;
    rgb_out[id] = clamp(c, 0.0f, 1.0f);
}
