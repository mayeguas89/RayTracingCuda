#pragma once
#include "cuda_runtime.h"
#include "vec3.h"

class Ray
{
public:
  __device__ Ray() {}
  __device__ Ray(const Vec3& origin, const Vec3& direction, float time = 0.f):
    origin_{origin},
    direction_{direction},
    time_{time}
  {}
  __device__ Vec3 origin() const
  {
    return origin_;
  }
  __device__ Vec3 direction() const
  {
    return direction_;
  }
  __device__ Vec3 at(float t) const
  {
    return origin_ + t * direction_;
  }

  __device__ float time() const
  {
    return time_;
  }

private:
  Vec3 origin_;
  Vec3 direction_;
  float time_;
};