#pragma once

#include "cuda_runtime.h"
#include "utils.h"

class Perlin
{
public:
  __device__ Perlin(curandState* state, float pattern_size): pattern_{pattern_size}, state_{state}
  {
    random_vec_ = new Vec3[kPointCount];
    for (int i = 0; i < kPointCount; i++)
      random_vec_[i] = getRandomVector(state, -1.f, 1.f);
    perm_x_ = perlinGeneratePerm();
    perm_y_ = perlinGeneratePerm();
    perm_z_ = perlinGeneratePerm();
  }
  ~Perlin()
  {
    delete[] random_vec_;
    delete[] perm_x_;
    delete[] perm_y_;
    delete[] perm_z_;
  }

  __device__ float turb(const Vec3& p, int depth = 7) const
  {
    auto accum = 0.f;
    auto temp_p = p;
    auto weight = 1.f;

    for (int i = 0; i < depth; i++)
    {
      accum += weight * noise(temp_p);
      weight *= 0.5f;
      temp_p *= 2.f;
    }

    return fabs(accum);
  }

  __device__ float noise(const Vec3& point) const
  {
    auto p = pattern_ * point;
    float u = p.x() - floor(p.x());
    float v = p.y() - floor(p.y());
    float w = p.z() - floor(p.z());

    // auto i = static_cast<int>(pattern_ * point.x()) & 255;
    // auto j = static_cast<int>(pattern_ * point.y()) & 255;
    // auto k = static_cast<int>(pattern_ * point.z()) & 255;
    // return random_float_[perm_x_[i] ^ perm_y_[j] ^ perm_z_[k]];

    int i = static_cast<int>(floor(p.x()));
    int j = static_cast<int>(floor(p.y()));
    int k = static_cast<int>(floor(p.z()));
    Vec3 c[2][2][2];
    for (int di = 0; di < 2; di++)
      for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
          c[di][dj][dk] = random_vec_[perm_x_[(i + di) & 255] ^ perm_y_[(j + dj) & 255] ^ perm_z_[(k + dk) & 255]];

    return trilinearInterp(c, u, v, w);
  }

private:
  static const int kPointCount = 256;

  __device__ void permute(int* p, int n)
  {
    for (int i = n - 1; i > 0; i--)
    {
      int target = (int)getRandom(state_, 0.f, (float)i);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }

  __device__ int* perlinGeneratePerm()
  {
    int* p = new int[kPointCount];
    for (int i = 0; i < kPointCount; i++)
      p[i] = i;
    permute(p, kPointCount);
    return p;
  }

  __device__ float trilinearInterp(Vec3 c[2][2][2], float u, float v, float w) const
  {
    float uu = u * u * (3.f - 2.f * u);
    float vv = v * v * (3.f - 2.f * v);
    float ww = w * w * (3.f - 2.f * w);
    float accum = 0.f;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
        {
          Vec3 weight_v(u - i, v - j, w - k);
          accum += (i * uu + (1.f - i) * (1.f - uu)) * (j * vv + (1.f - j) * (1.f - vv))
                   * (k * ww + (1.f - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
        }

    return accum;
  }

  float pattern_;
  curandState* state_;
  Vec3* random_vec_;
  int* perm_x_;
  int* perm_y_;
  int* perm_z_;
};