#include "br_model.h"

#include <cstdio>

#include "br_model_yolo.h"

bool br_model::init(const model_info& model_info)
{
    // Без выходов разбирать нечего: описатели собираются только по атрибутам модели
    if (model_info._output_count <= 0 || model_info._output_attrs == nullptr || model_info._output_mems == nullptr)
    {
        printf("%s: model has no outputs!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Данные лежат в памяти, привязанной к NPU, размерности и параметры квантования —
    // в атрибутах тензоров: из этого и получается описатель, которым пользуется разбор
    _tensors.clear();
    for (int i = 0; i < model_info._output_count; i++)
    {
        tensor output;
        output._data = static_cast<const int8_t*>(model_info._output_mems[i]->virt_addr);
        output._dims = model_info._output_attrs[i].dims;
        output._n_dims = model_info._output_attrs[i].n_dims;
        output._zp = model_info._output_attrs[i].zp;
        output._scale = model_info._output_attrs[i].scale;
        _tensors.push_back(output);
    }

    // Форму модели проверяет уже конкретная реализация: она одна знает, что должно
    // получиться на выходе
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
