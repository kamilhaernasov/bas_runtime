#include "br_model.h"

#include <cstdio>

#include "br_model_yolo.h"

std::unique_ptr<br_model> br_model_create(const std::string& type)
{
    if (type == "yolov5")
    {
        return std::make_unique<br_model_yolo>();
    }

    printf("%s: unknown model type '%s'!\n", __PRETTY_FUNCTION__, type.c_str());

    return nullptr;
}
