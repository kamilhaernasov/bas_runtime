#include "br_npu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

br_npu::br_npu()
{
    std::cout << __func__ << std::endl;
}

br_npu::~br_npu()
{
    std::cout << __func__ << std::endl;
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
    std::cout << __func__ << std::endl;

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
    ret = rknn_init(&_app_ctx._ctx, (void*)_ptr_model->get_model_path().c_str(), 0, 0, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_init fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Получаем количество входных и выходных тензоров модели
    rknn_input_output_num io_num;
    ret = rknn_query(_app_ctx._ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
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
        ret = rknn_query(_app_ctx._ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
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
        ret = rknn_query(_app_ctx._ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
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
        _app_ctx._ctx,
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
    ret = rknn_set_io_mem(_app_ctx._ctx, _app_ctx.input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("%s: input_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    printf("%s: input size_with_stride=%u\n", __PRETTY_FUNCTION__, input_attrs[0].size_with_stride);

    // Привязываем память выходных тензоров
    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        _app_ctx.output_mems[i] = rknn_create_mem(_app_ctx._ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(_app_ctx._ctx, _app_ctx.output_mems[i], &output_attrs[i]);
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

    // Отдаем форму модели на проверку разбору: он единственный, кто знает,
    // что должно получиться на выходе.
    // _input_width, _input_height — размер входа модели: по нему разбор считает шаг сетки.
    // _is_quant — выходы квантованы: разбор работает только с int8.
    // _outputs, _output_count — выходные тензоры: по их форме разбор выясняет число голов
    const std::vector<br_model::tensor> outputs = collect_outputs();
    const br_model::model_info model_info
    {
        ._input_width = _app_ctx.model_width,
        ._input_height = _app_ctx.model_height,
        ._input_channels = _app_ctx.model_channel,
        ._is_quant = _app_ctx.is_quant,
        ._outputs = outputs.data(),
        ._output_count = (int)outputs.size(),
    };
    if (!_ptr_model->init(model_info))
    {
        return false;
    }

    // Прогон модели и разбор ее выходов уходят в отдельный поток
    _worker = std::thread(&br_npu::worker, this);

    std::cout << "inited" << std::endl;

    return true;
}

bool br_npu::is_input_free()
{
    std::lock_guard<std::mutex> lock(_mtx);

    // Пока вход занят предыдущим кадром, кадр для детекции пропускается
    return _input_state == input_state::free;
}

void br_npu::submit_input()
{
    {
        std::lock_guard<std::mutex> lock(_mtx);
        if (_input_state != input_state::free)
        {
            return;
        }
        // Кадр уже уложен: прогон может стартовать
        _input_state = input_state::ready;
    }
    _cv.notify_one();
}

bool br_npu::run()
{
    int ret = 0;

    // Кадр в буфер пишет RGA, а читает NPU: синхронизируем кэш перед прогоном
    ret = rknn_mem_sync(_app_ctx._ctx, _app_ctx.input_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("%s: rknn_mem_sync fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    ret = rknn_run(_app_ctx._ctx, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_run fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

// Собирает выходные тензоры прогона в том виде, в котором их ждет разбор модели:
// данные лежат в памяти, привязанной rknn, а размерности и параметры квантования —
// в атрибутах тензора, которые получены при инициализации
std::vector<br_model::tensor> br_npu::collect_outputs() const
{
    std::vector<br_model::tensor> outputs;
    outputs.reserve(_app_ctx.output_mems.size());

    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        br_model::tensor output;
        output._data = static_cast<const int8_t*>(_app_ctx.output_mems[i]->virt_addr);
        output._dims = _app_ctx.output_attrs[i].dims;
        output._n_dims = _app_ctx.output_attrs[i].n_dims;
        output._zp = _app_ctx.output_attrs[i].zp;
        output._scale = _app_ctx.output_attrs[i].scale;
        outputs.push_back(output);
    }

    return outputs;
}

// После прогона отдаем выходные тензоры разбору модели. Готовые рамки складываем
// под мьютекс: цикл обработки кадров читает их оттуда, пока поток считает следующий кадр
bool br_npu::process_outputs()
{
    // Выходы прогона лежат в привязанной памяти: разбор читает их без копирования
    const std::vector<br_model::tensor> outputs = collect_outputs();

    br_model::detection_list detections;
    if (!_ptr_model->process(outputs.data(), (int)outputs.size(), detections))
    {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(_mtx_detections);
        _detections = std::move(detections);
        _inference_count++;
    }

    return true;
}

void br_npu::worker()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(_mtx);
            _cv.wait(lock, [this] { return _stop || _input_state == input_state::ready; });
            if (_stop)
            {
                return;
            }
            _input_state = input_state::busy;
        }

        if (!run() || !process_outputs())
        {
            printf("%s: frame skipped\n", __PRETTY_FUNCTION__);
        }

        {
            std::lock_guard<std::mutex> lock(_mtx);
            _input_state = input_state::free;
        }
        _cv.notify_all();
    }
}

br_model::detection_list br_npu::get_detections()
{
    // Копия под замком: цикл рисует по ней, пока поток считает следующий кадр
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
    // Останавливаем поток прогона до освобождения контекста и памяти
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _stop = true;
    }
    _cv.notify_all();
    if (_worker.joinable())
    {
        _worker.join();
    }

    // Отвязываем входные тензоры от блоков кадра (сами блоки и RGA-проход — за br_fsm и br_rga)
    for (size_t i = 0; i < _app_ctx.input_mems.size(); i++)
    {
        if (_app_ctx.input_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx._ctx, _app_ctx.input_mems[i]);
            _app_ctx.input_mems[i] = nullptr;
        }
    }
    _app_ctx.input_mems.clear();

    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        if (_app_ctx.output_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx._ctx, _app_ctx.output_mems[i]);
            _app_ctx.output_mems[i] = nullptr;
        }
    }
    _app_ctx.output_mems.clear();

    if (_app_ctx.input_attrs != nullptr)
    {
        free(_app_ctx.input_attrs);
        _app_ctx.input_attrs = nullptr;
    }

    if (_app_ctx.output_attrs != nullptr)
    {
        free(_app_ctx.output_attrs);
        _app_ctx.output_attrs = nullptr;
    }

    if (_app_ctx._ctx != 0)
    {
        rknn_destroy(_app_ctx._ctx);
        _app_ctx._ctx = 0;
    }

    return true;
}
