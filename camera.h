#pragma once

#include "ray.h"
#include "utils.h"
#include "vec3.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>

class Camera
{
public:
  Camera(const Vec3& look_from,
         const Vec3& look_at,
         const Vec3& up_world,
         float focus_distance,
         float aperture,
         float vertical_fov,
         float aspect_ratio,
         float time0 = 0.f,
         float time1 = 0.f):
    look_from_{look_from},
    look_at_{look_at},
    up_world_{up_world},
    focus_distance_{focus_distance},
    vertical_fov_{vertical_fov},
    aspect_ratio_{aspect_ratio},
    aperture_{aperture},
    time0_{time0},
    time1_{time1}
  {
    recalculateCamera();
  }

  __device__ Ray getRay(float u, float v, curandState* local_state)
  {
    Vec3 rd = lens_radius_ * getRandomInUnitDisk(local_state);
    Vec3 offset = horizontal_ * rd.x() + vertical_ * rd.y();

    Vec3 pixel = image_lower_left_corner_ + horizontal_ * u + vertical_ * v;
    Vec3 origin = (look_from_ + offset);
    Vec3 ray_dir = pixel - origin;
    float shutter_time = getRandom(local_state, time0_, time1_);
    // float shutter_time = (time1_ - time0_) / 2.f;
    return Ray(origin, ray_dir, shutter_time);
  }

  void zoomIn(float step)
  {
    vertical_fov_ -= step;
    recalculateCamera();
  }

  void zoomOut(float step)
  {
    vertical_fov_ += step;
    recalculateCamera();
  }

  void openAperture(float step)
  {
    aperture_ = std::clamp(aperture_ + step, 0.f, 16.f);
    recalculateCamera();
  }

  void closeAperture(float step)
  {
    aperture_ = std::clamp(aperture_ - step, 0.f, 16.f);
    recalculateCamera();
  }

  void rotateLeft(float step)
  {
    yaw_ = std::clamp(yaw_ + step, -360.f, 360.f);
    recalculateCamera();
  }

  void rotateRight(float step)
  {
    yaw_ = std::clamp(yaw_ - step, -360.f, 360.f);
    recalculateCamera();
  }

  void rotateUp(float step)
  {
    pitch_ = std::clamp(pitch_ - step, -360.f, 360.f);
    recalculateCamera();
  }

  void rotateDown(float step)
  {
    pitch_ = std::clamp(pitch_ + step, -360.f, 360.f);
    recalculateCamera();
  }

  void setData(Vec3 position, Vec3 look_at, float vfov, float aperture, float yaw, float pitch)
  {
    look_from_ = position;
    look_at_ = look_at;
    vertical_fov_ = vfov;
    aperture_ = aperture;
    yaw_ = yaw;
    pitch_ = pitch;
    recalculateCamera();
  }

private:
  void recalculateCamera()
  {
    focus_distance_ = (look_from_ - look_at_).length();
    float theta = deg2rad(vertical_fov_);
    float h_over_z = tanf(theta / 2.f);
    float viewport_height = 2.f * h_over_z;
    float viewport_width = aspect_ratio_ * viewport_height;

    float yaw = deg2rad(yaw_);
    float pitch = deg2rad(pitch_);
    glm::vec3 camera_target = {look_at_.x(), look_at_.y(), look_at_.z()};
    glm::vec3 camera_position = {look_from_.x(), look_from_.y(), look_from_.z()};
    glm::vec3 up = {up_world_.x(), up_world_.y(), up_world_.z()};
    glm::vec3 forward = glm::normalize(camera_position - camera_target);
    glm::vec3 right = glm::cross(forward, up);
    auto rot_pitch = glm::rotate(glm::mat4{1.0f}, pitch, right);
    auto rot_yaw = glm::rotate(glm::mat4{1.0f}, yaw, up);
    auto rot = rot_pitch * rot_yaw;
    camera_target = glm::vec3{glm::normalize(rot * glm::vec4(camera_target, 0.f))};

    auto c_f = unit_vector(Vec3(camera_position.x, camera_position.y, camera_position.z)
                           - Vec3(camera_target.x, camera_target.y, camera_target.z));
    auto c_r = unit_vector(cross(up_world_, c_f));
    Vec3 c_up = cross(c_f, c_r);
    horizontal_ = c_r * viewport_width;
    vertical_ = c_up * viewport_height;
    image_lower_left_corner_ = look_from_ - horizontal_ / 2.f - vertical_ / 2.f - focus_distance_ * c_f;
    lens_radius_ = aperture_ * .5f;
  }

public:
  Vec3 look_from_;
  Vec3 look_at_;
  Vec3 up_world_;
  Vec3 image_lower_left_corner_;
  Vec3 vertical_;
  Vec3 horizontal_;
  float lens_radius_;
  float aperture_;
  float focus_distance_;
  float vertical_fov_;
  float aspect_ratio_;
  float yaw_ = 0.f;
  float pitch_ = 0.f;
  // shutter open/close
  float time1_;
  float time0_;
};
