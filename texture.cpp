#include "texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

ImageTexture::ImageTexture(const char* filename)
{
  int components;
  pixels_ = stbi_load(filename, &width_, &height_, &components, kComponentsPerPixel);
  if (!pixels_)
  {
    std::cerr << "Error: could not load texture file " << filename << std::endl;
    width_ = height_ = 0;
  }
}