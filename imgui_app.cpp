#include "imgui_app.h"
// clang-format off
#include <GL/freeglut.h>
// clang-format on
#include "camera.h"
#include "texture.h"

#include <filesystem>

void ImguiApp::init()
{
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsLight();

  // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  // Setup Platform/Renderer backends
  // FIXME: Consider reworking this example to install our own GLUT funcs + forward calls ImGui_ImplGLUT_XXX ones, instead of using ImGui_ImplGLUT_InstallFuncs().
  ImGui_ImplGLUT_Init();
  ImGui_ImplOpenGL2_Init();
  // Install GLUT handlers (glutReshapeFunc(), glutMotionFunc(), glutPassiveMotionFunc(), glutMouseFunc(), glutKeyboardFunc() etc.)
  // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
  // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
  // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
  // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
  // io.WantCaptureKeyboard = true;
  // io.WantCaptureMouse = true;

  file_dialog_.SetTypeFilters({".jpg", ".png"});
}

void ImguiApp::setCamera(Camera* camera,
                         std::function<void(Vec3, Vec3, float, float, float, float)> camera_callback)
{
  camera_ = camera;
  camera_callback_ = camera_callback;
}

void ImguiApp::setTexture(unsigned int* texture_id, int w, int h)
{
  texture_id_ = texture_id;
  w_ = w;
  h_ = h;
}

void ImguiApp::registerGroundFunctions(std::function<void(float)> dielectric,
                                       std::function<void(Vec3, float)> metal,
                                       std::function<void(Vec3)> lambertian_solid_color,
                                       std::function<void(float, Vec3, Vec3)> lambertian_checker,
                                       std::function<void(float)> perlin_noise,
                                       std::function<void(float)> marble)
{
  change_ground_dielectric_callback_ = dielectric;
  change_ground_metal_callback_ = metal;
  change_ground_lambertian_solid_callback_ = lambertian_solid_color;
  change_ground_lambertian_checkboard_callback_ = lambertian_checker;
  change_ground_perlin_noise_callback_ = perlin_noise;
  change_ground_marble_callback_ = marble;
}

void ImguiApp::registerAddFunctions(std::function<void(Vec3, float, float)> dielectric,
                                    std::function<void(Vec3, float, Vec3, float)> metal,
                                    std::function<void(Vec3, float, Vec3)> lambertian_solid_color,
                                    std::function<void(Vec3, float, float, Vec3, Vec3)> lambertian_checker,
                                    std::function<void(Vec3, float, float)> perlin_noise,
                                    std::function<void(Vec3, float, float)> marble,
                                    std::function<void(Vec3, float, const char*)> image_texture,
                                    std::function<void(Vec3, float, Vec3, float)> diffuse_light)
{
  add_dielectric_callback_ = dielectric;
  add_metal_callback_ = metal;
  add_lambertian_solid_callback_ = lambertian_solid_color;
  add_lambertian_checkboard_callback_ = lambertian_checker;
  add_perlin_noise_callback_ = perlin_noise;
  add_marble_callback_ = marble;
  add_image_texture_ = image_texture;
  add_diffuse_light_callback_ = diffuse_light;
}

void ImguiApp::registerChangingBackground(std::function<void(Vec3, bool)> background)
{
  change_bg_callback_ = background;
}

void ImguiApp::ImGui_ImplGLUT_ReshapeFunc(int w, int h)
{
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)w, (float)h);
}

void ImguiApp::ImGui_ImplGLUT_MotionFunc(int x, int y)
{
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent((float)x, (float)y);
}

void ImguiApp::KeyboardFunc(unsigned char c, int x, int y)
{
  ImGui_ImplGLUT_KeyboardFunc(c, x, y);
}

void ImguiApp::KeyboardUpFunc(unsigned char c, int x, int y)
{
  ImGui_ImplGLUT_KeyboardUpFunc(c, x, y);
}

void ImguiApp::SpecialFunc(int key, int x, int y)
{
  ImGui_ImplGLUT_SpecialFunc(key, x, y);
}

void ImguiApp::SpecialUpFunc(int key, int x, int y)
{
  ImGui_ImplGLUT_SpecialUpFunc(key, x, y);
}

void ImguiApp::ImGui_ImplGLUT_MouseFunc(int glut_button, int state, int x, int y)
{
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent((float)x, (float)y);
  int button = -1;
  if (glut_button == GLUT_LEFT_BUTTON)
    button = 0;
  if (glut_button == GLUT_RIGHT_BUTTON)
    button = 1;
  if (glut_button == GLUT_MIDDLE_BUTTON)
    button = 2;
  if (button != -1 && (state == GLUT_DOWN || state == GLUT_UP))
    io.AddMouseButtonEvent(button, state == GLUT_DOWN);
}

void ImguiApp::step(GpuTimer* timer)
{
  // Start the Dear ImGui frame
  ImGui_ImplOpenGL2_NewFrame();
  ImGui_ImplGLUT_NewFrame();
  ImGui::NewFrame();
  ImGuiIO& io = ImGui::GetIO();

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Create the docking environment
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus
                                 | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

  ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("InvisibleWindow", nullptr, windowFlags);
  ImGui::PopStyleVar(3);

  ImGuiID dockSpaceId = ImGui::GetID("InvisibleWindowDockSpace");

  ImGui::DockSpace(dockSpaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
  ImGui::End();

  {
    ImGui::Begin("Scene");
    ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
    ImGui::Text("size = %d x %d", w_, h_);
    // ImGui::InputInt("Width", &w_);
    ImGui::SetCursorPos({(viewportPanelSize.x - w_) / 2.f, (viewportPanelSize.y - h_) / 2.f});
    ImGui::Image((void*)(intptr_t)(*texture_id_), ImVec2(w_, h_), ImVec2(0.f, 1.f), ImVec2(1.f, 0.f));
    ImGui::End();
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  ImGui::Begin("Configuration");
  {
    if (ImGui::CollapsingHeader("Ground"))
    {
      groundFactory();
    }
    if (ImGui::CollapsingHeader("Spheres"))
    {
      sphereFactory();
    }
    if (ImGui::CollapsingHeader("Background"))
    {
      background();
    }

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::Text("Last render took %f ms", timer->Elapsed());
  }
  ImGui::End();

  ImGui::Begin("Camera parameters");
  {
    cameraWidget();
  }
  ImGui::End();

  file_dialog_.Display();

  // Rendering
  ImGui::Render();
  ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }
}

void ImguiApp::destroy()
{
  // Cleanup
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGLUT_Shutdown();
  ImGui::DestroyContext();
}

void ImguiApp::cameraWidget()
{
  static float position[3] = {camera_->look_from_.x(), camera_->look_from_.y(), camera_->look_from_.z()};
  static float look_at[3] = {camera_->look_at_.x(), camera_->look_at_.y(), camera_->look_at_.z()};
  static float vertical_fov = camera_->vertical_fov_;
  static float aperture = camera_->aperture_;
  static float yaw = camera_->yaw_;
  static float pitch = camera_->pitch_;

  ImGui::InputFloat3("Position", position);
  ImGui::InputFloat3("Look At", look_at);
  ImGui::SliderFloat("Vertical FOV", &vertical_fov, 0.1f, 180.f);
  ImGui::SliderFloat("Aperture", &aperture, 0.0f, 20.f);
  ImGui::SliderFloat("Yaw", &yaw, -360.f, 360.f);
  ImGui::SliderFloat("Pitch", &pitch, -360.f, 360.f);

  if (ImGui::Button("Apply"))
  {
    camera_callback_(Vec3{position[0], position[1], position[2]},
                     Vec3{look_at[0], look_at[1], look_at[2]},
                     vertical_fov,
                     aperture,
                     yaw,
                     pitch);
  }
}

void ImguiApp::groundFactory()
{
  static ImVec4 solid_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static ImVec4 check_board_0 = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static ImVec4 check_board_1 = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  static ImVec4 metal_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static float metal_fuzzy = 0.0f;
  static float index_of_refraction = 1.5f;

  const char* materials[] = {"Lambertian", "Metal", "Dielectric"};
  const char* textures[] = {"CheckBoard", "SolidColor", "Perlin Noise", "Marble"};
  static int item_materials = 1;
  static int item_textures = 1;
  static float pattern_size = 5.f;

  ImGui::ListBox("Materials", &item_materials, materials, IM_ARRAYSIZE(materials), 3);
  if (item_materials == 0)
  {
    ImGui::ListBox("Textures", &item_textures, textures, IM_ARRAYSIZE(textures), 3);
    if (item_textures == 0)
    {
      ImGui::ColorEdit3("Checkboard Color 1", (float*)&check_board_0);
      ImGui::ColorEdit3("Checkboard Color 0", (float*)&check_board_1);
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);
      if (ImGui::Button("Apply"))
      {
        change_ground_lambertian_checkboard_callback_(pattern_size,
                                                      Vec3{check_board_0.x, check_board_0.y, check_board_0.z},
                                                      Vec3{check_board_1.x, check_board_1.y, check_board_1.z});
      }
    }
    else if (item_textures == 1)
    {
      ImGui::ColorEdit3("Solid Color", (float*)&solid_color);

      if (ImGui::Button("Apply"))
      {
        change_ground_lambertian_solid_callback_(Vec3{solid_color.x, solid_color.y, solid_color.z});
      }
    }
    else if (item_textures == 2)
    {
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);
      if (ImGui::Button("Apply"))
      {
        change_ground_perlin_noise_callback_(pattern_size);
      }
    }
    else if (item_textures == 3)
    {
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);
      if (ImGui::Button("Apply"))
      {
        change_ground_marble_callback_(pattern_size);
      }
    }
  }
  else if (item_materials == 1)
  {
    ImGui::ColorEdit3("Metal Color", (float*)&metal_color);
    ImGui::SliderFloat("Metal Fuzzy", &metal_fuzzy, 0.0f, 1.0f);

    if (ImGui::Button("Apply"))
    {
      change_ground_metal_callback_(Vec3{metal_color.x, metal_color.y, metal_color.z}, metal_fuzzy);
    }
  }
  else if (item_materials == 2)
  {
    ImGui::SliderFloat("Dielectric IR", &index_of_refraction, 0.0f, 3.f);

    if (ImGui::Button("Apply"))
    {
      change_ground_dielectric_callback_(index_of_refraction);
    }
  }
}

void ImguiApp::sphereFactory()
{
  static float position[3] = {0.f, 0.f, 0.f};
  static ImVec4 solid_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static ImVec4 check_board_0 = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static ImVec4 check_board_1 = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  static ImVec4 metal_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static float metal_fuzzy = 0.0f;
  static float index_of_refraction = 1.5f;
  static float light_intensity = 1.f;

  const char* materials[] = {"Lambertian", "Metal", "Dielectric"};
  const char* textures[] = {"CheckBoard", "SolidColor", "Perlin Noise", "Marble", "Image", "DiffuseLight"};
  static std::string filename = "";
  static int item_materials = 1;
  static int item_textures = 1;
  static float radius = 0.5f;
  static float pattern_size = 5.f;

  ImGui::InputFloat3("Position", position);
  ImGui::SliderFloat("Radius", &radius, 0.1f, 2.0f);
  ImGui::ListBox("Materials", &item_materials, materials, IM_ARRAYSIZE(materials), 3);
  if (item_materials == 0)
  {
    ImGui::ListBox("Textures", &item_textures, textures, IM_ARRAYSIZE(textures), 6);
    if (item_textures == 0)
    {
      ImGui::ColorEdit3("Checkboard Color 1", (float*)&check_board_0);
      ImGui::ColorEdit3("Checkboard Color 0", (float*)&check_board_1);
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);

      if (ImGui::Button("Add"))
      {
        add_lambertian_checkboard_callback_(Vec3{position[0], position[1], position[2]},
                                            radius,
                                            pattern_size,
                                            Vec3{check_board_0.x, check_board_0.y, check_board_0.z},
                                            Vec3{check_board_1.x, check_board_1.y, check_board_1.z});
      }
    }
    else if (item_textures == 1)
    {
      ImGui::ColorEdit3("Solid Color", (float*)&solid_color);

      if (ImGui::Button("Add"))
      {
        add_lambertian_solid_callback_(Vec3{position[0], position[1], position[2]},
                                       radius,
                                       Vec3{solid_color.x, solid_color.y, solid_color.z});
      }
    }
    else if (item_textures == 2)
    {
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);
      if (ImGui::Button("Add"))
      {
        add_perlin_noise_callback_(Vec3{position[0], position[1], position[2]}, radius, pattern_size);
      }
    }
    else if (item_textures == 3)
    {
      ImGui::SliderFloat("Pattern size", &pattern_size, 1.f, 20.f);
      if (ImGui::Button("Add"))
      {
        add_marble_callback_(Vec3{position[0], position[1], position[2]}, radius, pattern_size);
      }
    }
    else if (item_textures == 4)
    {
      if (ImGui::Button("Open image"))
        file_dialog_.Open();
      if (file_dialog_.HasSelected() && !file_dialog_.GetSelected().string().empty())
      {
        ImGui::Text("Filename: ");
        ImGui::SameLine();
        std::filesystem::path file_path{file_dialog_.GetSelected().string()};
        ImGui::Text(file_path.filename().string().c_str());

        int my_image_width = 0;
        int my_image_height = 0;
        unsigned int my_image_texture = 0;
        bool ret = LoadTextureFromFile(file_dialog_.GetSelected().string().c_str(),
                                       my_image_texture,
                                       my_image_width,
                                       my_image_height);
        IM_ASSERT(ret);
        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        static int w = 100;
        int h = w / (float)(my_image_width / (float)my_image_height);
        ImGui::Image((void*)(intptr_t)my_image_texture, ImVec2(w, h));

        if (ImGui::Button("Add"))
        {
          add_image_texture_(Vec3{position[0], position[1], position[2]},
                             radius,
                             file_dialog_.GetSelected().string().c_str());
        }
      }
    }
    else if (item_textures == 5)
    {
      ImGui::ColorEdit3("Color", (float*)&solid_color);
      ImGui::InputFloat("Light intensity", &light_intensity);

      if (ImGui::Button("Add"))
      {
        add_diffuse_light_callback_(Vec3{position[0], position[1], position[2]},
                                    radius,
                                    Vec3{solid_color.x, solid_color.y, solid_color.z},
                                    light_intensity);
      }
    }
  }
  else if (item_materials == 1)
  {
    ImGui::ColorEdit3("Metal Color", (float*)&metal_color);
    ImGui::SliderFloat("Metal Fuzzy", &metal_fuzzy, 0.0f, 1.0f);

    if (ImGui::Button("Add"))
    {
      add_metal_callback_(Vec3{position[0], position[1], position[2]},
                          radius,
                          Vec3{metal_color.x, metal_color.y, metal_color.z},
                          metal_fuzzy);
    }
  }
  else if (item_materials == 2)
  {
    ImGui::SliderFloat("Dielectric IR", &index_of_refraction, 0.0f, 3.f);

    if (ImGui::Button("Add"))
    {
      add_dielectric_callback_(Vec3{position[0], position[1], position[2]}, radius, index_of_refraction);
    }
  }
}

void ImguiApp::background()
{
  static ImVec4 color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static bool use_gradient = false;
  ImGui::ColorEdit3("Color", (float*)&color);
  ImGui::Checkbox("Use gradient", &use_gradient);

  if (ImGui::Button("Change"))
  {
    change_bg_callback_(Vec3{color.x, color.y, color.z}, use_gradient);
  }
}

// Simple helper function to load an image into a OpenGL texture with common settings
bool ImguiApp::LoadTextureFromFile(const char* filename,
                                   unsigned int& out_texture,
                                   int& out_width,
                                   int& out_height)
{
  ImageTexture texture(filename);
  out_width = texture.width_;
  out_height = texture.height_;

  // Create a OpenGL texture identifier
  glGenTextures(1, &out_texture);
  glBindTexture(GL_TEXTURE_2D, out_texture);

  // Setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGBA,
               texture.width_,
               texture.height_,
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               texture.pixels_);
  return true;
}