#pragma once
#include "hittable.h"

#include <memory>
#include <vector>

#define MAX_OBJECTS 50

class HittableList: public Hittable
{
private:
  Hittable** objects_;
  int num_objects_;

public:
  __device__ HittableList(Hittable** objects, int num_objects)
  {
    objects_ = objects;
    num_objects_ = num_objects;
  };

  __device__ void add(Hittable* object)
  {
    objects_[num_objects_++] = object;
  }

  __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override
  {
    auto closest = t_max;
    bool hit = false;

    for (int i = 0; i < num_objects_; i++)
    {
      HitRecord temp;
      if (objects_[i]->hit(r, t_min, closest, temp))
      {
        closest = temp.t;
        hit = true;
        rec = temp;
      }
    }

    return hit;
  }
};