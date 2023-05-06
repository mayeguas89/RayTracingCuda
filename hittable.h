#pragma once

#include "ray.h"

class Material;
struct HitRecord
{
  Vec3 p;
  Vec3 normal;
  float t;
  float u;
  float v;
  bool front_face;
  Material* material;

  __device__ HitRecord() {}

  __device__ void SetFace(const Ray& ray, const Vec3& out_normal)
  {
    front_face = dot(ray.direction(), out_normal) < 0.f;
    normal = (front_face) ? out_normal : -out_normal;
  }
};

class Hittable
{
public:
  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};