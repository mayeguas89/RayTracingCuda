#pragma once
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "vec3.h"

// Using reflect definition in https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
__device__ inline Vec3 reflect(const Vec3& incident, const Vec3& normal)
{
  return incident - 2 * dot(incident, normal) * normal;
}

// Using refract definition in https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
__device__ inline Vec3 refract(const Vec3& incident, const Vec3& normal, float eta)
{
  auto k = 1.0f - eta * eta * (1.0f - dot(normal, incident) * dot(normal, incident));
  if (k < 0.0)
    return {0.f, 0.f, 0.f};
  return eta * incident - (eta * dot(normal, incident) + sqrtf(k)) * normal;
}

__device__ inline float reflectance(float cosine, float ref_idx)
{
  // Use Schlick's approximation for reflectance.
  auto r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * static_cast<float>(pow((1.f - cosine), 5));
}

__device__ inline float getRandom(curandState* local_state)
{
  return curand_uniform(local_state);
}

__device__ inline float getRandom(curandState* local_state, float small, float big)
{
  float rand_0_1 = curand_uniform(local_state);
  float randf = rand_0_1 * (big - small + 0.999999f);
  randf += small;
  return randf;
}

__device__ inline Vec3 getRandomVector(curandState* local_state, float small, float big)
{
  return Vec3(getRandom(local_state, small, big),
              getRandom(local_state, small, big),
              getRandom(local_state, small, big));
}

// Returns vector with its three coordinates between 0.f and 1.f
__device__ inline Vec3 getRandomVector(curandState* local_state)
{
  return Vec3(curand_uniform(local_state), curand_uniform(local_state), curand_uniform(local_state));
}

// Returns a vector inside a sphere of unit radius
__device__ inline Vec3 getRandomInUnitSpehere(curandState* local_state)
{
  float r1 = curand_uniform(local_state);
  float r2 = curand_uniform(local_state);
  float phi = CURAND_2PI * r1;
  float theta = r2 * (float)CURAND_PI_DOUBLE / 2.f;
  float x = cosf(phi) * sinf(theta);
  float y = sinf(phi) * sinf(theta);
  float z = cosf(theta);
  return Vec3(x, y, z);
}

// Return a unit vector
__device__ inline Vec3 getRandomUnitVectorInUnitSpehere(curandState* local_state)
{
  Vec3 v = getRandomInUnitSpehere(local_state);
  v.make_unit_vector();
  return v;
}

// Return a unit vector in the same hemisphere that the normal
__device__ inline Vec3 getRandomUnitVectorInHemispehere(curandState* local_state, const Vec3& normal)
{
  float r1 = curand_uniform(local_state);
  float r2 = curand_uniform(local_state);
  float phi = CURAND_2PI * r1;
  float x = cosf(phi) * sqrtf(1.f - r2 * r2);
  float y = sinf(phi) * sqrtf(1.f - r2 * r2);
  float z = r2;
  Vec3 v(x, y, z);
  v.make_unit_vector();
  if (dot(normal, v) > .0f)
    return v;
  return -v;
}

__device__ inline Vec3 getRandomInUnitDisk(curandState* local_state)
{
  float x = getRandom(local_state) * 2.f - 1.f;
  float y = getRandom(local_state) * 2.f - 1.f;
  float z = 0.f;
  return unit_vector(Vec3{x, y, z});
}

__host__ __device__ inline float deg2rad(float rad)
{
  return rad * (float)CURAND_PI_DOUBLE / 180.f;
}