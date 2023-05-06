#pragma once
#include "cuda_runtime.h"
#include "ray.h"
#include "texture.h"
#include "utils.h"
#include "vec3.h"

#include <algorithm>

struct HitRecord;

class Material
{
public:
  __device__ virtual bool scatter(const Ray& ray,
                                  const HitRecord& hit_record,
                                  Vec3& attenuation,
                                  Ray& scattered,
                                  curandState* local_state) const = 0;
  __device__ virtual Vec3 emitted(float u, float v, const Vec3& point) const
  {
    return Vec3(0.f, 0.f, 0.f);
  }
};

class Lambertian: public Material
{
private:
  Texture* albedo_ = nullptr;

public:
  __device__ Lambertian(const Vec3& color): albedo_{new SolidColor(color)} {}
  __device__ Lambertian(Texture* texture): albedo_{texture} {}
  __device__ virtual bool scatter(const Ray& ray,
                                  const HitRecord& hit_record,
                                  Vec3& attenuation,
                                  Ray& scattered,
                                  curandState* local_state) const override
  {
    // Vec3 scatter_direction = hit_record.p + getRandomUnitVectorInHemispehere(local_state, hit_record.normal);
    Vec3 scatter_direction = hit_record.p + hit_record.normal + getRandomInUnitSpehere(local_state);
    if (scatter_direction.near_zero())
      scatter_direction = hit_record.normal;
    scattered = Ray(hit_record.p, scatter_direction, ray.time());
    float scatter_probability = clamp(getRandom(local_state) + 0.5f);
    attenuation = albedo_->value(hit_record.u, hit_record.v, hit_record.p);
    // attenuation = albedo_ / CURAND_2PI * 0.5f;
    return true;
  }
};

class Metal: public Material
{
private:
  Vec3 albedo_;
  float fuzzy_;

public:
  __device__ __host__ Metal(const Vec3& color, float fuzzy): albedo_{color}
  {
    fuzzy_ = fuzzy >= 0.f ? fuzzy : 1.f;
  }
  __device__ virtual bool scatter(const Ray& ray,
                                  const HitRecord& hit_record,
                                  Vec3& attenuation,
                                  Ray& scattered,
                                  curandState* local_state) const override
  {
    Vec3 reflected = reflect(unit_vector(ray.direction()), hit_record.normal);
    scattered = Ray(hit_record.p, reflected + fuzzy_ * getRandomInUnitSpehere(local_state), ray.time());
    float scatter_probability = getRandom(local_state);
    attenuation = albedo_;
    return dot(reflected, hit_record.normal) > 0.f;
  }
};

class Dielectric: public Material
{
private:
  float index_of_refraction_;

public:
  __device__ __host__ Dielectric(float index_of_refraction): index_of_refraction_{index_of_refraction} {}
  __device__ virtual bool scatter(const Ray& ray,
                                  const HitRecord& hit_record,
                                  Vec3& attenuation,
                                  Ray& scattered,
                                  curandState* local_state) const override
  {
    auto refr_ratio = hit_record.front_face ? (1.0f / index_of_refraction_) : index_of_refraction_;
    auto unit_ray = unit_vector(ray.direction());
    float cos_theta = fminf(dot(-unit_ray, hit_record.normal), 1.0f);
    auto sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    Vec3 direction;
    bool cannot_refract = refr_ratio * sin_theta > 1.f;
    // Randomly choose between fraction and reflection
    if (cannot_refract || reflectance(cos_theta, refr_ratio) > getRandom(local_state))
    {
      // Must reflect
      direction = reflect(unit_ray, hit_record.normal);
    }
    else
    {
      // can refract
      direction = refract(unit_ray, hit_record.normal, refr_ratio);
    }
    scattered = Ray(hit_record.p, direction, ray.time());
    // En material dielectrico reflejado es toda blanca
    attenuation = Vec3{1.f, 1.f, 1.f};
    return true;
  }
};

class DiffuseLight: public Material
{
private:
  Texture* emit_;

public:
  __device__ DiffuseLight(const Vec3& color): emit_{new SolidColor{color}} {}
  __device__ DiffuseLight(Texture* texture): emit_{texture} {}
  __device__ virtual bool scatter(const Ray& ray,
                                  const HitRecord& hit_record,
                                  Vec3& attenuation,
                                  Ray& scattered,
                                  curandState* local_state) const override
  {
    return false;
  }
  __device__ virtual Vec3 emitted(float u, float v, const Vec3& point) const override
  {
    return emit_->value(u, v, point);
  }
};