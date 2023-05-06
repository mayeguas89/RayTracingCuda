#pragma once
#include "cuda_runtime.h"
#include "hittable.h"
#include "ray.h"
#include "vec3.h"

class Material;

class Sphere: public Hittable
{
private:
  float radius_;
  Vec3 center_;
  Material* material_ = nullptr;

public:
  __device__ Sphere(const Vec3& center, float radius, Material* material):
    center_{center},
    radius_{radius},
    material_{material}
  {}

  __device__ void deleteMaterial()
  {
    delete material_;
  }

  __device__ void getSphereUV(const Vec3& point, float& u, float& v) const
  {
    // point: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    float theta = acosf(-point.y());
    float phi = atan2(-point.z(), point.x()) + (float)CURAND_PI_DOUBLE;
    u = phi / CURAND_2PI;
    v = theta / (float)CURAND_PI_DOUBLE;
  }

  __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override
  {
    Vec3 oc = ray.origin() - center_;
    float a = dot(ray.direction(), ray.direction());
    float b = dot(oc, ray.direction());
    float c = dot(oc, oc) - radius_ * radius_;

    float discr = (b * b) - (a * c);
    if (discr < 0.f)
      return false;
    float sqrtd = sqrtf(discr);
    float root = (-b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
      root = (-b + sqrtd) / a;
      if (root < t_min || t_max < root)
        return false;
    }
    rec.t = root;
    rec.p = ray.at(root);
    rec.material = material_;
    auto normal = (rec.p - center_) / radius_;
    rec.SetFace(ray, normal);
    getSphereUV(normal, rec.u, rec.v);
    return true;
  }
};

class MovingSphere: public Hittable
{
public:
  __device__ MovingSphere(const Vec3& center0,
                          const Vec3& center1,
                          float time0,
                          float time1,
                          float radius,
                          Material* material):
    center0_{center0},
    center1_{center1},
    time0_{time0},
    time1_{time1},
    radius_{radius},
    material_{material}
  {}
  __device__ Vec3 center(float t) const
  {
    auto c = ((t - time0_) / (time1_ - time0_)) * (center1_ - center0_);
    return center0_ + c;
  }

  __device__ void deleteMaterial()
  {
    delete material_;
  }

  __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override
  {
    Vec3 oc = ray.origin() - center(ray.time());
    float a = dot(ray.direction(), ray.direction());
    float b = dot(oc, ray.direction());
    float c = dot(oc, oc) - radius_ * radius_;

    float discr = (b * b) - (a * c);
    if (discr < 0.f)
      return false;
    float sqrtd = sqrtf(discr);
    float root = (-b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
      root = (-b + sqrtd) / a;
      if (root < t_min || t_max < root)
        return false;
    }
    rec.t = root;
    rec.p = ray.at(root);
    rec.material = material_;
    auto normal = (rec.p - center(ray.time())) / radius_;
    rec.SetFace(ray, normal);
    return true;
  }

private:
  float radius_;
  Vec3 center0_;
  Vec3 center1_;
  float time0_;
  float time1_;
  Material* material_ = nullptr;
};