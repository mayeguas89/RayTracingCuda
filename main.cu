#include "camera.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "hittable_list.h"
#include "imgui_app.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "utils.h"
#include "vec3.h"

#include <vector_types.h>

// clang-format off
#include <GL/glew.h>
#include <GL/freeglut.h>
// clang-format on

#include "timer.h"

#include <cuda_gl_interop.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <random>

#define THREADS_PER_BLOCK 8
#define NUM_SAMPLES 20
#define MAX_RAYS 50

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
__device__ Vec3
  getColor(const Ray& r, Hittable** world, curandState* local_state, bool use_gradient, Vec3 background)
{
  Vec3 cur_attenuation(1.f, 1.f, 1.f);
  Ray ray = r;
  for (int i = 0; i < MAX_RAYS; i++)
  {
    HitRecord hit_record;
    Vec3 emitted = {0.f, 0.f, 0.f};
    // 0.001f Fixing Shadow Acne
    if ((*world)->hit(ray, 0.001f, FLT_MAX, hit_record))
    {
      Ray scattered;
      Vec3 attenuation;
      emitted = hit_record.material->emitted(hit_record.u, hit_record.v, hit_record.p);

      if (hit_record.material->scatter(ray, hit_record, attenuation, scattered, local_state))
      {
        ray = scattered;
        cur_attenuation *= attenuation;
        // Ruleta rusa
        // if (getRandom(local_state) > 0.75f)
        // {
        //   float t = 0.5f * (unit_vector(ray.direction()).y() + 1.f);
        //   Vec3 f_color = Vec3(1.f, 1.f, 1.f) * (1.f - t) + Vec3(0.5f, 0.7f, 1.f) * t;
        //   return f_color * cur_attenuation / 0.75f;
        // }
      }
      else
        return emitted * cur_attenuation;
    }
    // If the ray hits nothing, return the background color.
    else
    {
      if (use_gradient)
      {
        float t = 0.5f * (unit_vector(ray.direction()).y() + 1.f);
        Vec3 f_color = Vec3(1.f, 1.f, 1.f) * (1.f - t) + background * t;
        return f_color * cur_attenuation;
      }
      return background * cur_attenuation;
    }
  }
  // After MAX_RAYS there is no light at all => dark
  return {0.f, 0.f, 0.f};
}

__global__ void createWorld(Hittable** world, Hittable** list, int size, curandState* local_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    // Material* ground = new Lambertian(new CheckerTexture(5.f, Vec3{0.2f, 0.3f, 0.1f}, Vec3{0.9f, 0.9f, 0.9f}));
    Material* ground = new Lambertian(new NoiseTexture(&local_state[0], 5.f));
    Material* center = new Lambertian(Vec3(0.7f, 0.3f, 0.3f));
    Material* left = new Metal(Vec3(0.8f, 0.8f, 0.8f), 0.f);
    Material* moving = new Lambertian(Vec3(0.8f, 0.5f, 0.2f));

    // Oro
    Material* right = new Metal(Vec3(1.f, 0.71f, 0.92f), .6f);
    Material* dielectric = new Dielectric(1.5f);
    Material* dielectric2 = new Dielectric(1.5f);

    list[0] = new MovingSphere{Vec3(0.75f, 1.f, -2.f), Vec3(0.75f, .75f, -2.f), 0.f, 2.f, 0.5f, moving};
    list[1] = new Sphere{Vec3(0.f, 0.f, -1.f), 0.5f, center};
    list[2] = new Sphere{Vec3(0.f, -100.5f, -1.f), 100.f, ground};
    list[3] = new Sphere{Vec3(-1.f, 0.f, -1.f), 0.5f, left};
    list[4] = new Sphere{Vec3(-0.75f, 0.f, 0.f), -0.5f, dielectric2};
    list[5] = new Sphere{Vec3(1.f, 0.f, -1.f), 0.5f, right};
    *world = new HittableList(list, size);
  }
}

__global__ void createRandomWorld(Hittable** world, Hittable** list, int size, curandState* d_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    auto ground_material = new Lambertian(Vec3(0.5f, 0.5f, 0.5f));
    list[size] = new Sphere{Vec3(0.f, -1000.f, 0.f), 1000.f, ground_material};

    auto material1 = new Dielectric(1.5);
    list[size + 1] = new Sphere{{0.f, 1.f, 0.f}, 1.f, material1};

    auto material2 = new Lambertian({0.4f, 0.2f, 0.1f});
    list[size + 2] = new Sphere{{-4.f, 1.f, 0.f}, 1.f, material2};

    auto material3 = new Metal({0.7f, 0.6f, 0.5f}, 0.0);
    list[size + 3] = new Sphere{{0.f, 1.f, 0.f}, 1.f, material3};
  }

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= size - 4)
    return;
  curandState local_state = d_state[x];
  auto choose_mat = getRandom(&local_state);
  auto center = Vec3(x + 0.9f * getRandom(&local_state), 0.2f, y + 0.9f * getRandom(&local_state));
  if ((center - Vec3(4.f, 0.2f, 0.f)).length() < 0.9)
  {
    if (choose_mat < 0.8f)
    {
      auto albedo = getRandomVector(&local_state) * getRandomVector(&local_state);
      auto material = new Lambertian(albedo);
      list[x] = new Sphere{center, 0.2f, material};
    }
    else if (choose_mat < 0.95f)
    {
      auto albedo = getRandomVector(&local_state) * getRandomVector(&local_state);
      auto fuzz = getRandom(&local_state) - 0.5f;
      fuzz = fuzz < 0.f ? 0.f : fuzz;
      auto material = new Metal(albedo, fuzz);
      list[x] = new Sphere{center, 0.2f, material};
    }
    else if (choose_mat < 0.95f)
    {
      auto material = new Dielectric(1.5f);
      list[x] = new Sphere{center, 0.2f, material};
    }
  }
  __syncthreads();
  *world = new HittableList(list, size);
}

__global__ void updateSpehereColor(int sphere_index, Hittable** list, curandState* state)
{
  Material* center = new Lambertian(getRandomVector(state) * getRandomVector(state));
  Sphere* s = reinterpret_cast<Sphere*>(list[sphere_index]);
  s->deleteMaterial();
  delete list[sphere_index];
  list[sphere_index] = new Sphere(Vec3(0.f, 0.f, getRandom(state, -2.f, 2.f)), 0.5f, center);
}

__global__ void updateGroundDielectric(Hittable** list, float ir)
{
  Material* ground = new Dielectric(ir);
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void updateGroundMetal(Hittable** list, Vec3 color, float fuzzy)
{
  Material* ground = new Metal(color, fuzzy);
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void updateGroundLambertianSolidColor(Hittable** list, Vec3 color)
{
  Material* ground = new Lambertian(color);
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void updateGroundLambertianCheckerTexture(Hittable** list, float pattern_size, Vec3 color1, Vec3 color2)
{
  Material* ground = new Lambertian(new CheckerTexture(pattern_size, color1, color2));
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void updateGroundLambertianPerlinNoiseTexture(Hittable** list, curandState* d_state, float pattern_size)
{
  Material* ground = new Lambertian(new NoiseTexture(d_state, pattern_size));
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void updateGroundLambertianMarbleTexture(Hittable** list, curandState* d_state, float pattern_size)
{
  Material* ground = new Lambertian(new MarbleTexture(d_state, pattern_size));
  Sphere* s = reinterpret_cast<Sphere*>(list[2]);
  s->deleteMaterial();
  delete list[2];
  list[2] = new Sphere(Vec3(0.f, -100.5f, -1.f), 100.f, ground);
}

__global__ void addDielectricSphere(Hittable** world, Vec3 position, float radius, float ir)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Dielectric(ir)));
}

__global__ void addMetalSphere(Hittable** world, Vec3 position, float radius, Vec3 color, float fuzzy)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Metal(color, fuzzy)));
}

__global__ void addLambertianSolidColorSphere(Hittable** world, Vec3 position, float radius, Vec3 color)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Lambertian(color)));
}

__global__ void addDiffuseLightSphere(Hittable** world, Vec3 position, float radius, Vec3 color, float intensity)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new DiffuseLight(intensity * color)));
}

__global__ void addLambertianCheckerTextureSphere(Hittable** world,
                                                  Vec3 position,
                                                  float radius,
                                                  float pattern_size,
                                                  Vec3 color1,
                                                  Vec3 color2)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Lambertian(new CheckerTexture(pattern_size, color1, color2))));
}

__global__ void addLambertianPerlinNoiseSphere(Hittable** world,
                                               curandState* d_state,
                                               Vec3 position,
                                               float radius,
                                               float pattern_size)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Lambertian(new NoiseTexture(d_state, pattern_size))));
}

__global__ void addLambertianMarbleSphere(Hittable** world,
                                          curandState* d_state,
                                          Vec3 position,
                                          float radius,
                                          float pattern_size)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Lambertian(new MarbleTexture(d_state, pattern_size))));
}

__global__ void addLambertianImageSphere(Hittable** world,
                                         Vec3 position,
                                         float radius,
                                         unsigned char* data,
                                         int width,
                                         int height)
{
  auto w = reinterpret_cast<HittableList*>(*world);
  w->add(new Sphere(position, radius, new Lambertian(new ImageTexture(data, width, height))));
}

__global__ void
  promediatePixels(uchar3* const pixel_color, int width, int height, cudaSurfaceObject_t viewCudaSurfaceObject)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int pixel_idx = y * width + x;
  uchar3 color = {0, 0, 0};
  Vec3 color3f;
  const float scale = 1.f / (float)NUM_SAMPLES;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    int index = i * width * height;
    auto tmp_color = pixel_color[pixel_idx + index];
    color3f += Vec3(scale * (float)(tmp_color.x / 255.f),
                    scale * (float)(tmp_color.y / 255.f),
                    scale * (float)(tmp_color.z / 255.f));
  }
  color = color3f.touchar3();
  uchar4 c4 = make_uchar4(color.x, color.y, color.z, 255);
  surf2Dwrite(c4, viewCudaSurfaceObject, x * sizeof(uchar4), y);
  pixel_color[pixel_idx] = color;
}

__global__ void render(uchar3* const pixel_color,
                       int width,
                       int height,
                       Camera* d_camera,
                       Hittable** world,
                       curandState* d_state,
                       cudaSurfaceObject_t viewCudaSurfaceObject,
                       Vec3 h_backgound,
                       bool use_gradient = false)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int image = blockDim.z * blockIdx.z + threadIdx.z;

  if (x >= width || y >= height || image >= NUM_SAMPLES)
    return;
  Vec3 color;
  int pixel_idx = y * width + x;
  int index = image * width * height;
  curandState local_state = d_state[index + pixel_idx];

  // float u = x / (float)NUM_SAMPLES;
  // float v = y / (float)NUM_SAMPLES;
  // u += getRandom(&local_state);
  // v += getRandom(&local_state);
  // u /= (float)(width - 1);
  // v /= (float)(height - 1);

  // for (int i = 0; i < NUM_SAMPLES; i++)
  // {
  float u = float((x + getRandom(&local_state))) / (float)(width - 1);
  float v = float((y + getRandom(&local_state))) / (float)(height - 1);

  Ray ray = d_camera->getRay(u, v, &local_state);
  color = getColor(ray, world, &local_state, use_gradient, h_backgound);
  // Scale color
  // const float scale = 1.f / (float)NUM_SAMPLES;
  // gamma-correct for gamma=2.0
  // tmp_color = Vec3(scale * tmp_color.x(), scale * tmp_color.y(), scale * tmp_color.z());
  //   color += tmp_color;
  // }

  // color = tmp_color;
  uchar3 c3 = color.touchar3();
  // uchar4 c4 = make_uchar4(c3.x, c3.y, c3.z, 255);
  // surf2Dwrite(c4, viewCudaSurfaceObject, x * sizeof(uchar4), y);
  pixel_color[index + pixel_idx] = c3;
}

__global__ void setupKernel(curandState* state, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int image = blockDim.z * blockIdx.z + threadIdx.z;
  if (x >= width || y >= height || image >= NUM_SAMPLES)
    return;
  int pixel_idx = y * width + x;
  int index = image * width * height;
  // Each thread gets same seed, different suquence, no offset
  curand_init(1234, index + pixel_idx, 0, &state[index + pixel_idx]);
}

__global__ void freeWorld(Hittable** world, Hittable** list, int size)
{
  MovingSphere* s = reinterpret_cast<MovingSphere*>(list[0]);
  s->deleteMaterial();
  delete list[0];
  for (int i = 1; i < size; i++)
  {
    Sphere* s = reinterpret_cast<Sphere*>(list[i]);
    s->deleteMaterial();
    delete list[i];
  }
  delete *world;
}

////////////////////////////////////////////////////////////////////////////////
// glut functions
////////////////////////////////////////////////////////////////////////////////
bool initGL(int* argc, char** argv);
void idle();
void display();
void render();
void reshape(int, int);
void mouseFunc(int, int, int, int);
void mouseMotionFunc(int, int);
void keyboardCallback(unsigned char key, int x, int y);
void keyboardUpCallback(unsigned char key, int x, int y);
void specialCallback(int key, int x, int y);
void specialUpCallback(int key, int x, int y);
void createTexture(GLuint* texture_id, struct cudaGraphicsResource** res, unsigned int res_flags);

////////////////////////////////////////////////////////////////////////////////
// constants
////////////////////////////////////////////////////////////////////////////////
const auto aspect_ratio = 16.f / 9.f;
const unsigned int window_width = 1200;
const unsigned int window_height = window_width / aspect_ratio;

////////////////////////////////////////////////////////////////////////////////
// variables
////////////////////////////////////////////////////////////////////////////////
GLuint texture_id;
struct cudaGraphicsResource* cuda_resource;
cudaSurfaceObject_t viewCudaSurfaceObject;
unsigned int threads_per_block;
bool has_to_render = true;
char* image_texture_data;

// cudaStream_t streams[NUM_SAMPLES];

////////////////////////////////////////////////////////////////////////////////
// world
////////////////////////////////////////////////////////////////////////////////
Hittable** d_world;
int world_size = 6;
Hittable** d_hittable;
uchar3* d_pixels;
uchar3* pixels;
curandState* d_states;
Vec3 h_background = {0.7f, 0.8f, 1.f};
bool h_use_gradient = false;

////////////////////////////////////////////////////////////////////////////////
// camera
////////////////////////////////////////////////////////////////////////////////
Camera* d_camera;
Camera* camera;

////////////////////////////////////////////////////////////////////////////////
// imgui
////////////////////////////////////////////////////////////////////////////////
ImguiApp imgui_app;

////////////////////////////////////////////////////////////////////////////////
// GpuTimer
////////////////////////////////////////////////////////////////////////////////
GpuTimer timer;

////////////////////////////////////////////////////////////////////////////////
//! Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  // Image
  int w = window_width;
  int h = window_height;
  float fov = 90.f;
  float viewport_height = 2.f;

  // Camera
  Vec3 look_at = Vec3(0.f, 0.f, -1.f);
  Vec3 look_from = Vec3(3.f, 3.f, 2.f);
  Vec3 up_world(0.f, 1.f, 0.f);
  float focus_distance = (look_at - look_from).length();
  float aperture = 0.f;
  camera = new Camera(look_from, look_at, up_world, focus_distance, aperture, fov, aspect_ratio, 0.f, 2.f);

  // GLUT
  initGL(&argc, argv);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  threads_per_block = powf(prop.maxThreadsPerBlock, 1 / 3.f);
  dim3 blockSize(threads_per_block, threads_per_block, threads_per_block);
  dim3 gridSize(ceil(w / (float)blockSize.x),
                ceil(h / (float)blockSize.y),
                ceil(NUM_SAMPLES / (float)blockSize.z));

  // for (int i = 0; i < NUM_SAMPLES; i++)
  //   cudaStreamCreate(&streams[i]);

  cudaMalloc((void**)&d_camera, sizeof(Camera));
  checkCudaErrors(cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice));

  // Randon numbers
  checkCudaErrors(cudaMalloc((void**)&d_states, sizeof(curandState) * w * h * NUM_SAMPLES));
  // for (int image = 0; image < NUM_SAMPLES; image++)
  //   setupKernel<<<gridSize, blockSize, 0, streams[image]>>>(d_states, w, h, image);
  setupKernel<<<gridSize, blockSize>>>(d_states, w, h);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // World
  checkCudaErrors(cudaMalloc((void**)&d_hittable, world_size * sizeof(Hittable)));
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable)));
  createWorld<<<1, 1>>>(d_world, d_hittable, world_size, d_states);
  // createRandomWorld<<<11, 11>>>(d_world, d_hittable, world_size, d_states);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Image pixels
  size_t num_bytes = w * h * sizeof(uchar3);
  pixels = (uchar3*)malloc(num_bytes);
  checkCudaErrors(cudaMalloc((void**)&d_pixels, num_bytes * NUM_SAMPLES));

  ////////////////////////////////////////////////////////////////////////////////
  // glut Main Loop
  ////////////////////////////////////////////////////////////////////////////////
  glutMainLoop();

  ////////////////////////////////////////////////////////////////////////////////
  // Write pixels to a file
  ////////////////////////////////////////////////////////////////////////////////
  cudaMemcpy(pixels, d_pixels, num_bytes, cudaMemcpyDeviceToHost);
  std::ofstream fileout("image.ppm");
  fileout << "P3\n";
  fileout << w << " " << h << std::endl;
  fileout << "255\n";
  for (int y = h - 1; y >= 0; --y)
  {
    for (int x = 0; x < w; x++)
    {
      fileout << pixels[y * w + x];
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Clean Up
  ////////////////////////////////////////////////////////////////////////////////
  imgui_app.destroy();

  freeWorld<<<1, 1>>>(d_world, d_hittable, world_size);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_pixels));
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_hittable));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_states));
  free(pixels);
  delete camera;
  cudaDeviceReset();
  return 0;
}

void updateSpehere()
{
  std::cout << "test" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int* argc, char** argv)
{
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(window_width * 3, window_height * 3);
  glutCreateWindow("Ray tracing");
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  glutIdleFunc(idle);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboardCallback);
  glutKeyboardUpFunc(keyboardUpCallback);
  glutSpecialFunc(specialCallback);
  glutSpecialUpFunc(specialUpCallback);
  glutMouseFunc(mouseFunc);
  glutMotionFunc(mouseMotionFunc);
  // initialize necessary OpenGL extensions
  glewInit();

  imgui_app.init();
  imgui_app.setTexture(&texture_id, window_width, window_height);
  imgui_app.setCamera(camera,
                      [&](Vec3 position, Vec3 look_at, float vfov, float aperture, float yaw, float pitch)
                      {
                        camera->setData(position, look_at, vfov, aperture, yaw, pitch);
                        cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
                        has_to_render = true;
                      });
  imgui_app.registerGroundFunctions(
    [&](float ir)
    {
      updateGroundDielectric<<<1, 1>>>(d_hittable, ir);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 color, float ir)
    {
      updateGroundMetal<<<1, 1>>>(d_hittable, color, ir);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 color)
    {
      updateGroundLambertianSolidColor<<<1, 1>>>(d_hittable, color);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](float pattern_size, Vec3 color1, Vec3 color2)
    {
      updateGroundLambertianCheckerTexture<<<1, 1>>>(d_hittable, pattern_size, color1, color2);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](float pattern_size)
    {
      updateGroundLambertianPerlinNoiseTexture<<<1, 1>>>(d_hittable, d_states, pattern_size);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](float pattern_size)
    {
      updateGroundLambertianMarbleTexture<<<1, 1>>>(d_hittable, d_states, pattern_size);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    });

  imgui_app.registerChangingBackground(
    [&](Vec3 color, bool use_gradient)
    {
      h_background = color;
      h_use_gradient = use_gradient;
      has_to_render = true;
    });

  imgui_app.registerAddFunctions(
    [&](Vec3 position, float radius, float ir)
    {
      world_size += 1;
      addDielectricSphere<<<1, 1>>>(d_world, position, radius, ir);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, Vec3 color, float fuzzy)
    {
      world_size += 1;
      addMetalSphere<<<1, 1>>>(d_world, position, radius, color, fuzzy);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, Vec3 color)
    {
      world_size += 1;
      addLambertianSolidColorSphere<<<1, 1>>>(d_world, position, radius, color);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, float pattern_size, Vec3 color1, Vec3 color2)
    {
      world_size += 1;
      addLambertianCheckerTextureSphere<<<1, 1>>>(d_world, position, radius, pattern_size, color1, color2);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, float pattern_size)
    {
      world_size += 1;
      addLambertianPerlinNoiseSphere<<<1, 1>>>(d_world, d_states, position, radius, pattern_size);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, float pattern_size)
    {
      world_size += 1;
      addLambertianMarbleSphere<<<1, 1>>>(d_world, d_states, position, radius, pattern_size);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, const char* filename)
    {
      world_size += 1;
      auto image_texture = ImageTexture(filename);
      unsigned char* d_texture_pixels;
      size_t pixels_size =
        image_texture.width_ * image_texture.height_ * ImageTexture::kComponentsPerPixel * sizeof(unsigned char);
      cudaMalloc((void**)&d_texture_pixels, pixels_size);
      cudaMemcpy(d_texture_pixels, image_texture.pixels_, pixels_size, cudaMemcpyHostToDevice);
      addLambertianImageSphere<<<1, 1>>>(d_world,
                                         position,
                                         radius,
                                         d_texture_pixels,
                                         image_texture.width_,
                                         image_texture.height_);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    },
    [&](Vec3 position, float radius, Vec3 color, float intensity)
    {
      world_size += 1;
      addDiffuseLightSphere<<<1, 1>>>(d_world, position, radius, color, intensity);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      has_to_render = true;
    });

  if (!glewIsSupported("GL_VERSION_2_0 "))
  {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! display callback
////////////////////////////////////////////////////////////////////////////////
void display() {}

////////////////////////////////////////////////////////////////////////////////
//! idle callback
////////////////////////////////////////////////////////////////////////////////
void idle()
{
  render();

  // glBindTexture(GL_TEXTURE_2D, texture_id);
  // {
  //   glBegin(GL_QUADS);
  //   {
  //     glTexCoord2f(0.0f, 0.0f);
  //     glVertex2f(-1.0f, -1.0f);
  //     glTexCoord2f(1.0f, 0.0f);
  //     glVertex2f(+1.0f, -1.0f);
  //     glTexCoord2f(1.0f, 1.0f);
  //     glVertex2f(+1.0f, +1.0f);
  //     glTexCoord2f(0.0f, 1.0f);
  //     glVertex2f(-1.0f, +1.0f);
  //   }
  //   glEnd();
  // }
  glBindTexture(GL_TEXTURE_2D, 0);
  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  imgui_app.step(&timer);

  glFinish();

  glutSwapBuffers();
  glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
//! Render Function
////////////////////////////////////////////////////////////////////////////////
void render()
{
  if (!has_to_render)
    return;

  ////////////////////////////////////////////////////////////////////////////////
  // cuda open gl interop
  ////////////////////////////////////////////////////////////////////////////////
  createTexture(&texture_id, &cuda_resource, cudaGraphicsMapFlagsWriteDiscard);
  cudaSurfaceObject_t viewCudaSurfaceObject;
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, 0));
  {
    cudaArray_t viewCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, cuda_resource, 0, 0);
    cudaResourceDesc viewCudaArrayResourceDesc;
    {
      viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
      viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
    }
    cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
    {
      // 3 dimensiones (x,y,imagen)
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);
      threads_per_block = powf(prop.maxThreadsPerBlock, 1 / 3.f);
      dim3 blockSize(threads_per_block, threads_per_block, threads_per_block);
      dim3 gridSize(ceil(window_width / (float)blockSize.x),
                    ceil(window_height / (float)blockSize.y),
                    ceil(NUM_SAMPLES / (float)blockSize.z));
      timer.Start();
      // Send a stream for each sample
      // for (int image = 0; image < NUM_SAMPLES; image++)
      // {
      // render<<<gridSize, blockSize, 0, streams[image]>>>(d_pixels,
      //                                                    window_width,
      //                                                    window_height,
      //                                                    d_camera,
      //                                                    d_world,
      //                                                    d_states,
      //                                                    viewCudaSurfaceObject,
      //                                                    h_background,
      //                                                    h_use_gradient,
      //                                                    image);
      render<<<gridSize, blockSize>>>(d_pixels,
                                      window_width,
                                      window_height,
                                      d_camera,
                                      d_world,
                                      d_states,
                                      viewCudaSurfaceObject,
                                      h_background,
                                      h_use_gradient);
      // }
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      // Promediate samples
      threads_per_block = sqrtf(prop.maxThreadsPerBlock);
      blockSize = dim3(threads_per_block, threads_per_block);
      gridSize = dim3(ceil(window_width / (float)blockSize.x), ceil(window_height / (float)blockSize.y));
      promediatePixels<<<gridSize, blockSize>>>(d_pixels, window_width, window_height, viewCudaSurfaceObject);
      timer.Stop();
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    cudaDestroySurfaceObject(viewCudaSurfaceObject);
  }
  cudaGraphicsUnmapResources(1, &cuda_resource);
  cudaStreamSynchronize(0);

  has_to_render = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////
void reshape(int x, int y)
{
  glViewport(0, 0, x, y);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

  imgui_app.ImGui_ImplGLUT_ReshapeFunc(x, y);
}

////////////////////////////////////////////////////////////////////////////////
//! mouse button callback
////////////////////////////////////////////////////////////////////////////////
void mouseFunc(int glut_button, int state, int x, int y)
{
  imgui_app.ImGui_ImplGLUT_MouseFunc(glut_button, state, x, y);
}

////////////////////////////////////////////////////////////////////////////////
//! mouse motion callback
////////////////////////////////////////////////////////////////////////////////
void mouseMotionFunc(int x, int y)
{
  imgui_app.ImGui_ImplGLUT_MotionFunc(x, y);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard callback
////////////////////////////////////////////////////////////////////////////////
void keyboardCallback(unsigned char key, int x, int y)
{
  imgui_app.KeyboardFunc(key, x, y);
  switch (key)
  {
    case 'z':
      camera->zoomIn(1.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case 'x':
      camera->zoomOut(1.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case 'a':
      camera->openAperture(.1f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case 's':
      camera->closeAperture(.1f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case 'd':
      updateSpehereColor<<<1, 1>>>(1, d_hittable, d_states);
      has_to_render = true;
      break;
  }
}

void keyboardUpCallback(unsigned char key, int x, int y)
{
  imgui_app.KeyboardUpFunc(key, x, y);
}

////////////////////////////////////////////////////////////////////////////////
//! special callback
////////////////////////////////////////////////////////////////////////////////
void specialUpCallback(int key, int x, int y)
{
  imgui_app.SpecialUpFunc(key, x, y);
}

void specialCallback(int key, int x, int y)
{
  imgui_app.SpecialFunc(key, x, y);
  switch (key)
  {
    case GLUT_KEY_LEFT:
      camera->rotateLeft(5.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case GLUT_KEY_RIGHT:
      camera->rotateRight(5.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case GLUT_KEY_UP:
      camera->rotateUp(5.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
    case GLUT_KEY_DOWN:
      camera->rotateDown(5.f);
      cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
      has_to_render = true;
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Create Texture
////////////////////////////////////////////////////////////////////////////////
void createTexture(GLuint* texture_id, struct cudaGraphicsResource** res, unsigned int res_flags)
{
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, texture_id);
  imgui_app.setTexture(texture_id, window_width, window_height);
  glBindTexture(GL_TEXTURE_2D, *texture_id);
  {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  }
  glBindTexture(GL_TEXTURE_2D, 0);

  checkCudaErrors(cudaGraphicsGLRegisterImage(res, *texture_id, GL_TEXTURE_2D, res_flags));
}
