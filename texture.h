#pragma once

#include "noise.h"
#include "vec3.h"

class Texture
{
public:
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const = 0;
};

class SolidColor: public Texture
{
public:
  __device__ SolidColor() {}
  __device__ SolidColor(const Vec3& color): value_{color} {}
  __device__ SolidColor(float r, float g, float b): value_{Vec3{r, g, b}} {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override
  {
    return value_;
  }

private:
  Vec3 value_;
};

class CheckerTexture: public Texture
{
public:
  __device__ CheckerTexture() {}
  __device__ CheckerTexture(float pattern, Texture* even, Texture* odd): pattern_{pattern}, even_{even}, odd_{odd}
  {}
  __device__ CheckerTexture(float pattern, const Vec3& color_odd, const Vec3& color_even):
    pattern_{pattern},
    even_{new SolidColor(color_even)},
    odd_{new SolidColor(color_odd)}
  {}

  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override
  {
    auto sines = sin(pattern_ * p.x()) * sin(pattern_ * p.y()) * sin(pattern_ * p.z());
    if (sines < 0.f)
      return odd_->value(u, v, p);
    return even_->value(u, v, p);
  }

private:
  Texture* odd_;
  Texture* even_;
  float pattern_;
};

class NoiseTexture: public Texture
{
public:
  __device__ NoiseTexture(curandState* state, float pattern): noise_{state, pattern}, pattern_{pattern} {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override
  {
    // return Vec3{1.f, 1.f, 1.f} * 0.5f * (1.f + sin(p.z() * pattern_ + 10.f * noise_.turb(p)));
    return Vec3{1.f, 1.f, 1.f} * noise_.turb(p);
  }

private:
  Perlin noise_;
  float pattern_;
};

class MarbleTexture: public Texture
{
public:
  __device__ MarbleTexture(curandState* state, float pattern): noise_{state, pattern}, pattern_{pattern} {}
  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override
  {
    return Vec3{1.f, 1.f, 1.f} * 0.5f * (1.f + sin(p.z() * pattern_ + 10.f * noise_.turb(p)));
  }

private:
  Perlin noise_;
  float pattern_;
};

class ImageTexture: public Texture
{
public:
  __device__ ImageTexture(unsigned char* data, int w, int h): pixels_{data}, width_{w}, height_{h} {}
  ImageTexture(const char* filename);

  ~ImageTexture()
  {
    delete pixels_;
  }

  __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override
  {
    if (!pixels_)
      return Vec3{0.3f, 1.f, 1.f};

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = clamp(u);
    v = 1.f - clamp(v); // Flip V to image coordinates

    auto i = static_cast<int>(u * width_);
    auto j = static_cast<int>(v * height_);

    // Clamp integer mapping, since actual coordinates should be less than 1.0
    if (i >= width_)
      i = width_ - 1;
    if (j >= height_)
      j = height_ - 1;

    const float color_scale = 1.f / 255.f;
    // Start at 0
    auto pixel = pixels_;
    // Go to the row
    pixel += j * (kComponentsPerPixel * width_);
    // Go to the column
    pixel += i * kComponentsPerPixel;

    return Vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
  }

  unsigned char* GetPixels()
  {
    return pixels_;
  }

  inline static const int kComponentsPerPixel = 4;
  unsigned char* pixels_;
  int width_;
  int height_;
};