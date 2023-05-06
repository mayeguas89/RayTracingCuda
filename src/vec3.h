#pragma once

#include "cuda_runtime.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

__host__ __device__ inline int toInt(float x);
class Vec3
{
public:
  __host__ __device__ Vec3(float x = 0.f, float y = 0.f, float z = 0.f)
  {
    x_ = x;
    y_ = y;
    z_ = z;
  }
  __host__ __device__ inline float x() const
  {
    return x_;
  }
  __host__ __device__ inline float y() const
  {
    return y_;
  }
  __host__ __device__ inline float z() const
  {
    return z_;
  }

  __host__ __device__ inline const Vec3& operator+() const
  {
    return *this;
  }
  __host__ __device__ inline Vec3 operator-() const
  {
    return Vec3(-x_, -y_, -z_);
  }

  __host__ __device__ inline Vec3& operator+=(const Vec3& other);
  __host__ __device__ inline Vec3& operator-=(const Vec3& other);
  __host__ __device__ inline Vec3& operator*=(const Vec3& other);
  __host__ __device__ inline Vec3& operator/=(const Vec3& other);
  __host__ __device__ inline Vec3& operator*=(const float k);
  __host__ __device__ inline Vec3& operator/=(const float k);

  __host__ __device__ inline bool near_zero() const
  {
    return (fabs(x_) < FLT_EPSILON) && (fabs(y_) < FLT_EPSILON) && (fabs(z_) < FLT_EPSILON);
  }

  __host__ __device__ inline float length() const
  {
    return sqrt(x_ * x_ + y_ * y_ + z_ * z_);
  }
  __host__ __device__ inline float squared_length() const
  {
    return x_ * x_ + y_ * y_ + z_ * z_;
  }
  __host__ __device__ inline void make_unit_vector();

  __device__ inline uchar3 touchar3()
  {
    return make_uchar3(toInt(x_), toInt(y_), toInt(z_));
  }

private:
  float x_, y_, z_;
};

__host__ __device__ inline void Vec3::make_unit_vector()
{
  float k = 1.0 / sqrt(x_ * x_ + y_ * y_ + z_ * z_);
  x_ *= k;
  y_ *= k;
  z_ *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2)
{
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
{
  return Vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
{
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2)
{
  return Vec3(v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z());
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v)
{
  return Vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t)
{
  return Vec3(v.x() / t, v.y() / t, v.z() / t);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t)
{
  return Vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2)
{
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2)
{
  return Vec3((v1.y() * v2.z() - v1.z() * v2.y()),
              (-(v1.x() * v2.z() - v1.z() * v2.x())),
              (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v)
{
  x_ += v.x_;
  y_ += v.y_;
  z_ += v.z_;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v)
{
  x_ *= v.x_;
  y_ *= v.y_;
  z_ *= v.z_;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v)
{
  x_ /= v.x_;
  y_ /= v.y_;
  z_ /= v.z_;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v)
{
  x_ -= v.x_;
  y_ -= v.y_;
  z_ -= v.z_;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t)
{
  x_ *= t;
  y_ *= t;
  z_ *= t;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t)
{
  float k = 1.0 / t;

  x_ *= k;
  y_ *= k;
  z_ *= k;
  return *this;
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v)
{
  return v / v.length();
}

__host__ __device__ inline float clamp(float x)
{
  return (x < 0.f) ? 0.f : (x > 1.f) ? 1.f : x;
}

__host__ __device__ inline int toInt(float x)
{
  return static_cast<int>(clamp(x) * 255.f);
}

inline std::ostream& operator<<(std::ostream& out, Vec3 c)
{
  out << toInt(c.x()) << " " << toInt(c.y()) << " " << toInt(c.z()) << std::endl;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, uchar3 c)
{
  out << (int)c.x << " " << (int)c.y << " " << (int)c.z << std::endl;
  return out;
}