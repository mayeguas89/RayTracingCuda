#pragma once
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include "vec3.h"
// clang-format off
#include "imfilebrowser.h"
// clang-format on

#include "timer.h"

#include <functional>
#include <string>
#include <unordered_map>

struct Camera;
class ImguiApp
{
private:
  inline static bool show_demo_window = true;
  inline static bool show_another_window = false;
  inline static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

public:
  void init();

  void setCamera(Camera* camera, std::function<void(Vec3, Vec3, float, float, float, float)> camera_callback);

  void setTexture(unsigned int* texture_id, int w, int h);

  void registerGroundFunctions(std::function<void(float)>,
                               std::function<void(Vec3, float)>,
                               std::function<void(Vec3)>,
                               std::function<void(float, Vec3, Vec3)>,
                               std::function<void(float)>,
                               std::function<void(float)>);

  void registerAddFunctions(std::function<void(Vec3, float, float)>,
                            std::function<void(Vec3, float, Vec3, float)>,
                            std::function<void(Vec3, float, Vec3)>,
                            std::function<void(Vec3, float, float, Vec3, Vec3)>,
                            std::function<void(Vec3, float, float)>,
                            std::function<void(Vec3, float, float)>,
                            std::function<void(Vec3, float, const char*)>,
                            std::function<void(Vec3, float, Vec3, float)>);

  void registerChangingBackground(std::function<void(Vec3, bool)>);

  void ImGui_ImplGLUT_ReshapeFunc(int w, int h);

  void ImGui_ImplGLUT_MotionFunc(int x, int y);

  void KeyboardFunc(unsigned char c, int x, int y);

  void KeyboardUpFunc(unsigned char c, int x, int y);

  void SpecialFunc(int key, int x, int y);

  void SpecialUpFunc(int key, int x, int y);

  void ImGui_ImplGLUT_MouseFunc(int glut_button, int state, int x, int y);

  void step(GpuTimer* timer);

  void destroy();

private:
  void sphereFactory();
  void groundFactory();
  void cameraWidget();
  void background();
  bool LoadTextureFromFile(const char* filename, unsigned int& out_texture, int& out_width, int& out_height);

  ////////////////////////////////////////////////////////////////////////////////
  //! Camera
  ////////////////////////////////////////////////////////////////////////////////
  Camera* camera_;
  std::function<void(Vec3, Vec3, float, float, float, float)> camera_callback_;

  ////////////////////////////////////////////////////////////////////////////////
  //! Texture
  ////////////////////////////////////////////////////////////////////////////////
  unsigned int* texture_id_;
  int w_;
  int h_;

  ////////////////////////////////////////////////////////////////////////////////
  //! File Browser
  ////////////////////////////////////////////////////////////////////////////////
  ImGui::FileBrowser file_dialog_;

  ////////////////////////////////////////////////////////////////////////////////
  //! Ground callback
  ////////////////////////////////////////////////////////////////////////////////
  std::function<void(float)> change_ground_dielectric_callback_;
  std::function<void(Vec3, float)> change_ground_metal_callback_;
  std::function<void(Vec3)> change_ground_lambertian_solid_callback_;
  std::function<void(float, Vec3, Vec3)> change_ground_lambertian_checkboard_callback_;
  std::function<void(float)> change_ground_perlin_noise_callback_;
  std::function<void(float)> change_ground_marble_callback_;

  ////////////////////////////////////////////////////////////////////////////////
  //! Add callback
  ////////////////////////////////////////////////////////////////////////////////
  std::function<void(Vec3, float, float)> add_dielectric_callback_;
  std::function<void(Vec3, float, Vec3, float)> add_metal_callback_;
  std::function<void(Vec3, float, Vec3)> add_lambertian_solid_callback_;
  std::function<void(Vec3, float, float, Vec3, Vec3)> add_lambertian_checkboard_callback_;
  std::function<void(Vec3, float, float)> add_perlin_noise_callback_;
  std::function<void(Vec3, float, float)> add_marble_callback_;
  std::function<void(Vec3, float, const char*)> add_image_texture_;
  std::function<void(Vec3, float, Vec3, float)> add_diffuse_light_callback_;

  std::function<void(Vec3, bool)> change_bg_callback_;
};