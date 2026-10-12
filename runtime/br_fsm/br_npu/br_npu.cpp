#include "br_npu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// Ниже порт postprocess.cc из примера luckfox_pico_rtsp_yolov5: деквантование выходных
// тензоров, разбор боксов и NMS. Наружу из модуля ничего не торчит, все внутренности
// доступны только методам br_npu.
namespace
{
    // Сколько значений выходного тензора приходится на один бокс помимо оценок классов:
    // четыре координаты и уверенность объекта. Это формат головы YOLOv5, а не свойство
    // модели: тензор отдает только общее число каналов, где в них кончаются координаты,
    // знает только сам алгоритм разбора (см. индексы hw_ptr ниже)
    constexpr int BOX_AND_CONF_SIZE = 5;

    // Названия классов и якоря голов — свойства конкретной модели, поэтому лежат
    // в файлах рядом с ней и читаются в init
    std::vector<std::string> labels;
    std::vector<float> anchors;

    int clamp_value(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

    int32_t clip_value(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        return (int8_t)clip_value(dst_val, -128, 127);
    }

    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

    // Файл меток: по названию класса в строке, число строк задает число классов модели
    bool load_labels(const std::string& path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            printf("%s: open %s fail!\n", __PRETTY_FUNCTION__, path.c_str());
            return false;
        }

        labels.clear();
        std::string line;
        while (std::getline(file, line))
        {
            // В файле могут быть виндовые переводы строк
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (!line.empty())
            {
                labels.push_back(line);
            }
        }

        printf("%s: %s: %u classes\n", __PRETTY_FUNCTION__, path.c_str(), (unsigned)labels.size());

        return !labels.empty();
    }

    // Файл якорей: числа парами (ширина, высота), по паре на якорь.
    // Идут подряд по головам: сначала все якоря старшей головы, потом следующей
    bool load_anchors(const std::string& path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            printf("%s: open %s fail!\n", __PRETTY_FUNCTION__, path.c_str());
            return false;
        }

        anchors.clear();
        std::string line;
        while (std::getline(file, line))
        {
            const char* cursor = line.c_str();
            while (*cursor != '\0')
            {
                char* end = nullptr;
                const double value = strtod(cursor, &end);
                if (end == cursor)
                {
                    break;
                }
                anchors.push_back(static_cast<float>(value));
                cursor = end;
            }
        }

        printf("%s: %s: %u values\n", __PRETTY_FUNCTION__, path.c_str(), (unsigned)anchors.size());

        return !anchors.empty();
    }

    float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0,
                            float xmin1, float ymin1, float xmax1, float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    int nms(const int valid_count, const std::vector<float>& output_locations, const std::vector<int>& class_ids,
            std::vector<int>& order, const int filter_id, const float threshold)
    {
        for (int i = 0; i < valid_count; ++i)
        {
            if (order[i] == -1 || class_ids[i] != filter_id)
            {
                continue;
            }
            int n = order[i];
            for (int j = i + 1; j < valid_count; ++j)
            {
                int m = order[j];
                if (m == -1 || class_ids[i] != filter_id)
                {
                    continue;
                }
                float xmin0 = output_locations[n * 4 + 0];
                float ymin0 = output_locations[n * 4 + 1];
                float xmax0 = output_locations[n * 4 + 0] + output_locations[n * 4 + 2];
                float ymax0 = output_locations[n * 4 + 1] + output_locations[n * 4 + 3];

                float xmin1 = output_locations[m * 4 + 0];
                float ymin1 = output_locations[m * 4 + 1];
                float xmax1 = output_locations[m * 4 + 0] + output_locations[m * 4 + 2];
                float ymax1 = output_locations[m * 4 + 1] + output_locations[m * 4 + 3];

                float iou = calculate_overlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold)
                {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }

    int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
    {
        float key;
        int key_index;
        int low = left;
        int high = right;
        if (left < right)
        {
            key_index = indices[left];
            key = input[left];
            while (low < high)
            {
                while (low < high && input[high] <= key)
                {
                    high--;
                }
                input[low] = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key)
                {
                    low++;
                }
                input[high] = input[low];
                indices[high] = indices[low];
            }
            input[low] = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }

}

br_npu::br_npu()
{

}

br_npu::~br_npu()
{

}

void br_npu::dump_tensor_attr(const rknn_tensor_attr* ptr_attr)
{
    printf("  index=%u, name=%s, n_dims=%u, dims=[%u, %u, %u, %u], n_elems=%u, size=%u, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           ptr_attr->index, ptr_attr->name, ptr_attr->n_dims, ptr_attr->dims[0], ptr_attr->dims[1], ptr_attr->dims[2], ptr_attr->dims[3],
           ptr_attr->n_elems, ptr_attr->size, get_format_string(ptr_attr->fmt), get_type_string(ptr_attr->type),
           get_qnt_type_string(ptr_attr->qnt_type), ptr_attr->zp, ptr_attr->scale);
}

bool br_npu::init(MB_BLK ptr_blk)
{
    int ret = 0;

    // Кадр во вход укладывает RGA, а боксы возвращаются в координаты кадра по его геометрии
    if (_settings._scale <= 0)
    {
        printf("%s: letterbox scale is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Сколько рамок помещается в кадр, задается настройкой: при нуле детекций не будет
    if (_settings._max_detections <= 0)
    {
        printf("%s: max detections is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Создаем контекст NPU и загружаем модель
    ret = rknn_init(&_ctx, (void*)_settings._model_path.c_str(), 0, 0, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_init fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Получаем количество входных и выходных тензоров модели
    rknn_input_output_num io_num;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("%s: rknn_query in_out_num fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Получаем атрибуты входных тензоров
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));

    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query input_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Получаем атрибуты выходных тензоров
    // В zero-copy режиме запрашиваются native-атрибуты выходных тензоров
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query output_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Габариты входа модели
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

    // Названия классов и якоря голов лежат в файлах рядом с моделью: обе вещи относятся
    // именно к ней, поэтому задаются данными, а не константами
    if (!load_labels(_settings._path_to_labels) || !load_anchors(_settings._path_to_anchors))
    {
        return false;
    }

    // Тензоров в контексте столько, сколько их у модели, и число голов — это число
    // выходных тензоров: все числа берутся из атрибутов, под них же резервируется память
    _app_ctx.io_num = io_num;
    _app_ctx.anchor_branch_num = io_num.n_output;
    if (_app_ctx.anchor_branch_num <= 0)
    {
        printf("%s: model has no outputs!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Вход модели один: кадр в него укладывает RGA, привязать второй вход не к чему
    if (io_num.n_input != 1)
    {
        printf("%s: model inputs %d is not supported!\n", __PRETTY_FUNCTION__, io_num.n_input);
        return false;
    }
    _app_ctx.input_mems.assign(io_num.n_input, nullptr);
    _app_ctx.output_mems.assign(_app_ctx.anchor_branch_num, nullptr);

    // Якоря лежат парами (ширина, высота) и поделены между головами поровну
    const int anchor_count = (int)anchors.size() / 2;
    if (anchor_count % _app_ctx.anchor_branch_num != 0)
    {
        printf("%s: anchors %d do not match heads %d!\n",
               __PRETTY_FUNCTION__, anchor_count, _app_ctx.anchor_branch_num);
        return false;
    }
    _app_ctx.anchor_per_branch = anchor_count / _app_ctx.anchor_branch_num;

    // Каналы выхода: на каждую ячейку сетки по prop_box_size значений на каждый якорь
    const int channels = (int)output_attrs[0].dims[output_attrs[0].n_dims - 1];
    if (channels % _app_ctx.anchor_per_branch != 0)
    {
        printf("%s: output channels %d do not match anchors per head %d!\n",
               __PRETTY_FUNCTION__, channels, _app_ctx.anchor_per_branch);
        return false;
    }
    _app_ctx.prop_box_size = channels / _app_ctx.anchor_per_branch;

    // Классов столько, сколько строк в файле меток: имена классов и выход модели
    // описывают одну модель, поэтому их числа должны сойтись
    _app_ctx.class_num = (int)labels.size();
    if (_app_ctx.prop_box_size != _app_ctx.class_num + BOX_AND_CONF_SIZE)
    {
        printf("%s: classes mismatch: model %d, %s %d!\n",
               __PRETTY_FUNCTION__, _app_ctx.prop_box_size - BOX_AND_CONF_SIZE,
               _settings._path_to_labels.c_str(), _app_ctx.class_num);
        return false;
    }

    printf("%s: heads=%d, anchors per head=%d, box size=%d, classes=%d\n",
           __PRETTY_FUNCTION__, _app_ctx.anchor_branch_num, _app_ctx.anchor_per_branch,
           _app_ctx.prop_box_size, _app_ctx.class_num);

    // По умолчанию тип входа int8 (нормализация и квантование считаются снаружи модели).
    // Uint8 позволяет совместить нормализацию и квантование с NPU
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // По умолчанию формат NHWC: NPU 1106 в zero-copy режиме поддерживает только его
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;

    // Буфер под вход модели выделен пулом br_mb_pool: проверяем, что его хватает под тензор
    const uint64_t size_blk = RK_MPI_MB_GetSize(ptr_blk);
    if (size_blk < input_attrs[0].size_with_stride)
    {
        printf("%s: block too small! need=%u, have=%llu\n",
               __PRETTY_FUNCTION__, input_attrs[0].size_with_stride, (unsigned long long)size_blk);
        return false;
    }

    // Привязываем блок к входному тензору модели.
    // rknn_create_mem_from_mb_blk в librknnmrt этой версии не экспортируется (есть только
    // в заголовке), поэтому отдаем блок через его dma-buf дескриптор.
    _app_ctx.input_mems[0] = rknn_create_mem_from_fd(
        _ctx,
        RK_MPI_MB_Handle2Fd(ptr_blk),
        RK_MPI_MB_Handle2VirAddr(ptr_blk),
        input_attrs[0].size_with_stride,
        0
    );
    if (_app_ctx.input_mems[0] == nullptr)
    {
        printf("%s: rknn_create_mem_from_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    ret = rknn_set_io_mem(_ctx, _app_ctx.input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("%s: input_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    printf("%s: input size_with_stride=%u\n", __PRETTY_FUNCTION__, input_attrs[0].size_with_stride);

    // Привязываем память выходных тензоров
    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        _app_ctx.output_mems[i] = rknn_create_mem(_ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(_ctx, _app_ctx.output_mems[i], &output_attrs[i]);
        if (ret < 0)
        {
            printf("%s: output_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
    }

    // Заполняем контекст модели
    _app_ctx.rknn_ctx = _ctx;

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        _app_ctx.is_quant = true;
    }
    else
    {
        _app_ctx.is_quant = false;
    }

    _app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    _app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // Прогон модели и разбор ее выходов уходят в отдельный поток
    _worker = std::thread(&br_npu::worker, this);

    printf("%s: ok\n", __PRETTY_FUNCTION__);

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
    ret = rknn_mem_sync(_ctx, _app_ctx.input_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("%s: rknn_mem_sync fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    ret = rknn_run(_ctx, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_run fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

bool br_npu::process_outputs()
{
    object_detect_result_list detections;

    if (!post_process(detections))
    {
        return false;
    }

    if (!map_detections_to_frame(detections))
    {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(_mtx_detections);
        _detections = detections;
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

// Разбор одной головы модели: раскладка NHWC, деквантование и декодирование боксов.
// Число классов, размер описания бокса и якоря берутся из контекста модели, а не из
// констант: в выходном тензоре на каждую ячейку сетки идет anchor_per_branch якорей
// по prop_box_size значений
int br_npu::process_head(const int head, std::vector<float>& boxes, std::vector<float>& box_scores,
                         std::vector<int>& class_ids)
{
    const int grid_h = _app_ctx.output_attrs[head].dims[2];
    const int grid_w = _app_ctx.output_attrs[head].dims[1];
    const int stride = _app_ctx.model_height / grid_h;
    const int32_t zp = _app_ctx.output_attrs[head].zp;
    const float scale = _app_ctx.output_attrs[head].scale;
    const int8_t* input = (int8_t*)(_app_ctx.output_mems[head]->virt_addr);
    const float threshold = _settings._box_thresh;
    const int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    const int align_c = _app_ctx.prop_box_size * _app_ctx.anchor_per_branch;
    const float* ptr_anchor = &anchors.at(head * _app_ctx.anchor_per_branch * 2);

    int valid_count = 0;

    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < _app_ctx.anchor_per_branch; a++)
            {
                const int hw_offset = h * grid_w * align_c + w * align_c + a * _app_ctx.prop_box_size;
                const int8_t* hw_ptr = input + hw_offset;
                const int8_t box_confidence = hw_ptr[4];

                if (box_confidence >= thres_i8)
                {
                    int8_t max_class_probs = hw_ptr[BOX_AND_CONF_SIZE];
                    int max_class_id = 0;
                    for (int k = 1; k < _app_ctx.class_num; ++k)
                    {
                        const int8_t prob = hw_ptr[BOX_AND_CONF_SIZE + k];
                        if (prob > max_class_probs)
                        {
                            max_class_id = k;
                            max_class_probs = prob;
                        }
                    }

                    const float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    const float class_prob_f32 = deqnt_affine_to_f32(max_class_probs, zp, scale);
                    const float limit_score = box_conf_f32 * class_prob_f32;

                    if (limit_score > threshold)
                    {
                        float box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0 - 0.5;
                        float box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0 - 0.5;
                        float box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0;
                        float box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;

                        box_x = (box_x + w) * (float)stride;
                        box_y = (box_y + h) * (float)stride;
                        box_w *= ptr_anchor[a * 2];
                        box_h *= ptr_anchor[a * 2 + 1];

                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                        box_scores.push_back(limit_score);
                        class_ids.push_back(max_class_id);
                        valid_count++;
                    }
                }
            }
        }
    }

    return valid_count;
}

bool br_npu::post_process(object_detect_result_list& detections)
{
    std::vector<float> filter_boxes;
    std::vector<float> obj_probs;
    std::vector<int> class_ids;
    int valid_count = 0;

    detections.id = 0;
    detections.count = 0;
    // Размер списка задает настройка, он же ограничивает число детекций в кадре
    detections.results.assign(_settings._max_detections, object_detect_result {});

    for (int i = 0; i < _app_ctx.anchor_branch_num; i++)
    {
        // RV1106 отдает выходные тензоры в NHWC и только в i8
        if (_app_ctx.is_quant)
        {
            valid_count += process_head(i, filter_boxes, obj_probs, class_ids);
        }
    }

    // Ничего не нашли
    if (valid_count <= 0)
    {
        return true;
    }

    std::vector<int> index_array;
    for (int i = 0; i < valid_count; i++)
    {
        index_array.push_back(i);
    }
    quick_sort_indice_inverse(obj_probs, 0, valid_count - 1, index_array);

    std::set<int> class_set(std::begin(class_ids), std::end(class_ids));
    for (auto c : class_set)
    {
        nms(valid_count, filter_boxes, class_ids, index_array, c, _settings._nms_thresh);
    }

    int last_count = 0;
    for (int i = 0; i < valid_count; i++)
    {
        if (index_array[i] == -1 || last_count >= _settings._max_detections)
        {
            continue;
        }

        const int n = index_array[i];
        const float x1 = filter_boxes[n * 4 + 0];
        const float y1 = filter_boxes[n * 4 + 1];
        const float x2 = x1 + filter_boxes[n * 4 + 2];
        const float y2 = y1 + filter_boxes[n * 4 + 3];

        detections.results[last_count].box.left   = clamp_value(x1, 0, _app_ctx.model_width);
        detections.results[last_count].box.top    = clamp_value(y1, 0, _app_ctx.model_height);
        detections.results[last_count].box.right  = clamp_value(x2, 0, _app_ctx.model_width);
        detections.results[last_count].box.bottom = clamp_value(y2, 0, _app_ctx.model_height);
        detections.results[last_count].prop = obj_probs[i];
        detections.results[last_count].cls_id = class_ids[n];
        last_count++;
    }
    detections.count = last_count;

    return true;
}

bool br_npu::map_detections_to_frame(object_detect_result_list& detections)
{
    for (int i = 0; i < detections.count; i++)
    {
        object_detect_result* ptr_result = &(detections.results[i]);

        // Переводим боксы из пространства входа модели в координаты кадра
        ptr_result->box.left   = clamp_value(((float)ptr_result->box.left   - _settings._pad_x) / _settings._scale, 0, _settings._width - 1);
        ptr_result->box.top    = clamp_value(((float)ptr_result->box.top    - _settings._pad_y) / _settings._scale, 0, _settings._height - 1);
        ptr_result->box.right  = clamp_value(((float)ptr_result->box.right  - _settings._pad_x) / _settings._scale, 0, _settings._width - 1);
        ptr_result->box.bottom = clamp_value(((float)ptr_result->box.bottom - _settings._pad_y) / _settings._scale, 0, _settings._height - 1);
    }

    return true;
}

br_npu::object_detect_result_list br_npu::get_detections()
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

const char* br_npu::get_class_name(const int cls_id)
{
    static char null_str[] = "null";

    if (cls_id < 0 || cls_id >= (int)labels.size())
    {
        return null_str;
    }

    return labels[cls_id].c_str();
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
            rknn_destroy_mem(_ctx, _app_ctx.input_mems[i]);
            _app_ctx.input_mems[i] = nullptr;
        }
    }
    _app_ctx.input_mems.clear();

    for (size_t i = 0; i < _app_ctx.output_mems.size(); i++)
    {
        if (_app_ctx.output_mems[i] != nullptr)
        {
            rknn_destroy_mem(_ctx, _app_ctx.output_mems[i]);
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

    if (_ctx != 0)
    {
        rknn_destroy(_ctx);
        _ctx = 0;
    }

    // Значения из файлов больше не нужны: контекст модели уже разобран
    labels.clear();
    anchors.clear();

    return true;
}

void br_npu::set_settings(const settings settings)
{
    _settings = settings;
}

