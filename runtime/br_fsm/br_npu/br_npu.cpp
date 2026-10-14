#include "br_npu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

br_npu::br_npu()
{
    
}

br_npu::~br_npu()
{
    
}

void br_npu::dump_tensor_attr(const rknn_tensor_attr* ptr_attr)
{
    printf("index=%u, name=%s, n_dims=%u, dims=[%u, %u, %u, %u], n_elems=%u, size=%u, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           ptr_attr->index, ptr_attr->name, ptr_attr->n_dims, ptr_attr->dims[0], ptr_attr->dims[1], ptr_attr->dims[2], ptr_attr->dims[3],
           ptr_attr->n_elems, ptr_attr->size, get_format_string(ptr_attr->fmt), get_type_string(ptr_attr->type),
           get_qnt_type_string(ptr_attr->qnt_type), ptr_attr->zp, ptr_attr->scale);
}

bool br_npu::init(MB_BLK ptr_blk, br_model* ptr_model)
{
    printf("%s: init start\n", __PRETTY_FUNCTION__);

    int ret = 0;

    // Модели нет? Ну, выходим
    if (ptr_model == nullptr)
    {
        printf("%s: ptr_model is null!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Модель получили, радостно...
    _ptr_model = ptr_model;

    // Достаем из модели разрешение входа
    const uint16_t width_model = _ptr_model->get_input_width();
    const uint16_t height_model = _ptr_model->get_input_height();

    // Создаем контекст NPU и загружаем модель
    ret = rknn_init(&_app_ctx.rknn_ctx, (void*)_ptr_model->get_model_path().c_str(), 0, 0, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_init fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Получаем количество входных и выходных тензоров модели
    rknn_input_output_num io_num;
    ret = rknn_query(_app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("%s: rknn_query in_out_num fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Получаем атрибуты входных тензоров
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    // Входных тензоров может быть несколько, достаем атрибуты всех
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(_app_ctx.rknn_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query input_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Получаем атрибуты выходных тензоров
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    // Выходных тензоров может быть (даже будет) несколько, достаем атрибуты всех
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(_app_ctx.rknn_ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query output_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // TODO: пока только для yolo с одним входным тензором: кадром
    if (io_num.n_input != 1)
    {
        printf("%s: model inputs %d is not supported!\n", __PRETTY_FUNCTION__, io_num.n_input);
        return false;
    }

    // Достаем, собственно, размер входа от модели
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("%s: model is NCHW input fmt\n", __PRETTY_FUNCTION__);
        _app_ctx.model_channel = input_attrs[0].dims[1];
        _app_ctx.model_height  = input_attrs[0].dims[2];
        _app_ctx.model_width   = input_attrs[0].dims[3];
    }
    else
    {
        printf("%s: model is NHWC input fmt\n", __PRETTY_FUNCTION__);
        _app_ctx.model_height  = input_attrs[0].dims[1];
        _app_ctx.model_width   = input_attrs[0].dims[2];
        _app_ctx.model_channel = input_attrs[0].dims[3];
    }

    printf("%s: model input height=%d, width=%d, channel=%d\n",
           __PRETTY_FUNCTION__, _app_ctx.model_height, _app_ctx.model_width, _app_ctx.model_channel);

    // Если у нас вход по факту от модели не совпадает с настройками
    if (_app_ctx.model_width != width_model || _app_ctx.model_height != height_model)
    {
        printf("%s: model input %dx%d, but settings say %ux%u!\n",
               __PRETTY_FUNCTION__, _app_ctx.model_width, _app_ctx.model_height,
               width_model, height_model);
        return false;
    }

    // Выходных тензоров не может быть 0 или меньше
    if (io_num.n_output <= 0)
    {
        printf("%s: model has no outputs!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Резервируем память под тензоры на вход и выход модели
    _app_ctx.io_num = io_num;
    _app_ctx.input_mems.assign(io_num.n_input, nullptr);
    _app_ctx.output_mems.assign(io_num.n_output, nullptr);

    // TODO: пока только для yolo (тама один тензор на вход, 3 на выход)
    // Устанавливаем тип входного тензора
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // Устанавливаем формат входного тензора (zero-copy) работает только в NHWC
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;

    // Проверяем блок памяти под вход модели, что влезаем
    const uint64_t size_blk = RK_MPI_MB_GetSize(ptr_blk);
    if (size_blk < input_attrs[0].size_with_stride)
    {
        printf("%s: block too small! need=%u, have=%llu\n",
               __PRETTY_FUNCTION__, input_attrs[0].size_with_stride, (unsigned long long)size_blk);
        return false;
    }

    // Привязываем блок памяти к входному тензору модели.
    // Брать будет сеть на вход из этого блока!
    _app_ctx.input_mems[0] = rknn_create_mem_from_fd(
        _app_ctx.rknn_ctx,
        RK_MPI_MB_Handle2Fd(ptr_blk),
        RK_MPI_MB_Handle2VirAddr(ptr_blk),
        input_attrs[0].size_with_stride,
        0
    );
    // Если не получилось привязать блок:
    if (_app_ctx.input_mems[0] == nullptr)
    {
        printf("%s: rknn_create_mem_from_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Теперь передаем в rknn наш входной блок
    ret = rknn_set_io_mem(_app_ctx.rknn_ctx, _app_ctx.input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("%s: input_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    printf("%s: input size_with_stride=%u\n", __PRETTY_FUNCTION__, input_attrs[0].size_with_stride);

    // Привязываем память выходных тензоров
    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        _app_ctx.output_mems[i] = rknn_create_mem(_app_ctx.rknn_ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(_app_ctx.rknn_ctx, _app_ctx.output_mems[i], &output_attrs[i]);
        if (ret < 0)
        {
            printf("%s: output_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
    }

    // Что-то про квантование: дернул из примера из SDK
    // UPD: это флаг, есть ли квантование на выходе модели. Вроде.
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        _app_ctx.is_quant = true;
    }
    else
    {
        _app_ctx.is_quant = false;
    }

    // Копируем атрибуты тензоров в контекст модели
    _app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    _app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // Собираем сопроводительную информацию для модели
    const br_model::model_info model_info
    {
        ._input_width = _app_ctx.model_width, // Ширина входа модели по факту от самой модели
        ._input_height = _app_ctx.model_height, // Высота входа модели по факту от самой модели
        ._input_channels = _app_ctx.model_channel, // Кол-во каналов во входе модели
        ._is_quant = _app_ctx.is_quant, // Флаг, что модель по факту квантованная
        ._output_attrs = _app_ctx.output_attrs, // Атрибуты выходных тензоров модели
        ._output_mems = _app_ctx.output_mems.data(), // Память выходных тензоров модели
        ._output_count = _app_ctx.output_mems.size(), // Кол-во выходных тензоров модели
    };
    // Пытаемся инициализировать модель
    if (!_ptr_model->init(model_info))
    {
        printf("%s: can't init model!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Прогон модели и разбор ее выходов уходят в отдельный поток
    _worker = std::thread(&br_npu::worker, this);

    printf("%s: init done\n", __PRETTY_FUNCTION__);

    return true;
}

bool br_npu::is_input_free()
{
    std::lock_guard<std::mutex> lock(_mtx);
    return _input_state == input_state::free;
}

void br_npu::submit_input()
{
    {
        std::lock_guard<std::mutex> lock(_mtx);
        // Коли мы не были свободны, значит произошла какая-то фигня
        if (_input_state != input_state::free)
        {
            return;
        }
        // Готовы стартовать
        _input_state = input_state::ready;
    }
    _cv.notify_one();
}

bool br_npu::run()
{
    int ret = 0;

    // Синхронизация кэша между CPU и NPU
    ret = rknn_mem_sync(_app_ctx.rknn_ctx, _app_ctx.input_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("%s: rknn_mem_sync fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }
    // Запускаем инференс
    ret = rknn_run(_app_ctx.rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_run fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

bool br_npu::process_outputs()
{
    // Список детекций
    br_model::detection_list detections;
    // Собираем из выходных тензоров модели детекции
    if (!_ptr_model->process(detections))
    {
        printf("%s: model process fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    {
        // Обновляем детекции на выдачу и счетчик успешных прогонов кадров
        std::lock_guard<std::mutex> lock(_mtx_detections);
        _detections = std::move(detections);
        _inference_count++;
    }

    return true;
}

void br_npu::worker()
{
    // Поток инференса
    while (true)
    {
        {
            // Ждем stop или готовность входа
            std::unique_lock<std::mutex> lock(_mtx);
            _cv.wait(lock, [this] { return _stop || _input_state == input_state::ready; });
            if (_stop)
            {
                return;
            }
            // Помечаем, что мы заняты прогоном кадра
            _input_state = input_state::busy;
        }

        // Запускаем инференс, а после обработку детекций
        if (!run() || !process_outputs())
        {
            printf("%s: frame skipped\n", __PRETTY_FUNCTION__);
        }

        {
            std::lock_guard<std::mutex> lock(_mtx);
            // Помечаем, что мы свободны и готовы принять следующий кадр
            _input_state = input_state::free;
        }
        _cv.notify_all();
    }
}

br_model::detection_list br_npu::get_detections()
{
    std::lock_guard<std::mutex> lock(_mtx_detections);
    return _detections;
}

uint32_t br_npu::get_inference_count()
{
    std::lock_guard<std::mutex> lock(_mtx_detections);
    return _inference_count;
}

bool br_npu::release()
{
    // Стопаем поток инференса
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _stop = true;
    }
    _cv.notify_all();
    // Ждем завершения потока
    if (_worker.joinable())
    {
        _worker.join();
    }

    // Уведомляем NPU, что мы разбираем память под вход
    for (size_t i = 0; i < _app_ctx.input_mems.size(); i++)
    {
        if (_app_ctx.input_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx.rknn_ctx, _app_ctx.input_mems[i]);
            _app_ctx.input_mems[i] = nullptr;
        }
    }
    // Чистим вектор входных тензоров
    _app_ctx.input_mems.clear();

    // Уведомляем NPU, что мы разбираем память под выход
    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        if (_app_ctx.output_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx.rknn_ctx, _app_ctx.output_mems[i]);
            _app_ctx.output_mems[i] = nullptr;
        }
    }
    // Чистим вектор выходных тензоров
    _app_ctx.output_mems.clear();

    // Чистим атрибуты тензоров входных
    if (_app_ctx.input_attrs != nullptr)
    {
        free(_app_ctx.input_attrs);
        _app_ctx.input_attrs = nullptr;
    }

    // Чистим атрибуты тензоров выходных
    if (_app_ctx.output_attrs != nullptr)
    {
        free(_app_ctx.output_attrs);
        _app_ctx.output_attrs = nullptr;
    }

    // Собственно, уничтожаем контекст модели
    if (_app_ctx.rknn_ctx != 0)
    {
        rknn_destroy(_app_ctx.rknn_ctx);
        _app_ctx.rknn_ctx = 0;
    }

    return true;
}
