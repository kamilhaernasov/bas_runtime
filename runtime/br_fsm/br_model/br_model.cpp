#include "br_model.h"

#include <cstdio>

#include "br_model_yolo.h"

bool br_model::init(const model_info& model_info)
{
    // Если нам не передали инфу о выходных тензорах, зачем вообще продолжать?
    if (model_info._output_count <= 0 || model_info._output_attrs == nullptr || model_info._output_mems == nullptr)
    {
        printf("%s: model has no outputs!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Заполняем выходные тензоры.
    _tensors.clear();
    for (int i = 0; i < model_info._output_count; i++)
    {
        tensor output;
        output._data = static_cast<const int8_t*>(model_info._output_mems[i]->virt_addr); // Указатель на данные тензора.
        output._dims = model_info._output_attrs[i].dims; // Атрибуты тензора.
        output._n_dims = model_info._output_attrs[i].n_dims; // Размерность тензора.
        output._zp = model_info._output_attrs[i].zp; // zero-point тензора.
        output._scale = model_info._output_attrs[i].scale; // scale тензора.
        _tensors.push_back(output);
    }

    // Стартуем внутреннию реализацию инициализации.
    return setup(model_info);
}

std::unique_ptr<br_model> br_model_create(const std::string& type)
{
    if (type == "yolov5")
    {
        return std::make_unique<br_model_yolo>();
    }

    printf("%s: unknown model type '%s'!\n", __PRETTY_FUNCTION__, type.c_str());

    return nullptr;
}
