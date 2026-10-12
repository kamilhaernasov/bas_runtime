#include "br_npu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

// Ниже порт postprocess.cc из примера luckfox_pico_rtsp_yolov5: деквантование выходных
// тензоров, разбор боксов и NMS. Наружу из модуля ничего не торчит, все внутренности
// доступны только методам br_npu.
namespace
{
    // Путь до файла с названиями классов модели
    constexpr const char* LABEL_NAME_TXT_PATH = "./model/coco_80_labels_list.txt";
    constexpr int OBJ_CLASS_NUM = 80;
    constexpr int PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM;
    constexpr int ANCHOR_PER_BRANCH = 3;
    constexpr int ANCHOR_BRANCH_NUM = 3;

    const int anchor[ANCHOR_BRANCH_NUM][6] = {{10, 13, 16, 30, 33, 23},
                                              {30, 61, 62, 45, 59, 119},
                                              {116, 90, 156, 198, 373, 326}};

    char* labels[OBJ_CLASS_NUM] = {};

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

    char* read_line(FILE* fp, char* buffer, int* len)
    {
        int ch;
        int i = 0;
        size_t buff_len = 0;

        buffer = (char*)malloc(buff_len + 1);
        if (!buffer)
        {
            return nullptr;
        }

        while ((ch = fgetc(fp)) != '\n' && ch != EOF)
        {
            buff_len++;
            void* tmp = realloc(buffer, buff_len + 1);
            if (tmp == nullptr)
            {
                free(buffer);
                return nullptr;
            }
            buffer = (char*)tmp;

            buffer[i] = (char)ch;
            i++;
        }
        buffer[i] = '\0';

        *len = buff_len;

        // Дошли до конца файла
        if (ch == EOF && (i == 0 || ferror(fp)))
        {
            free(buffer);
            return nullptr;
        }
        return buffer;
    }

    int read_lines(const char* file_name, char* lines[], int max_line)
    {
        FILE* file = fopen(file_name, "r");
        char* s = nullptr;
        int i = 0;
        int n = 0;

        if (file == nullptr)
        {
            printf("Open %s fail!\n", file_name);
            return -1;
        }

        while ((s = read_line(file, s, &n)) != nullptr)
        {
            lines[i++] = s;
            if (i >= max_line)
            {
                break;
            }
        }
        fclose(file);
        return i;
    }

    int load_label_name(const char* location_filename, char* label[])
    {
        printf("load lable %s\n", location_filename);
        read_lines(location_filename, label, OBJ_CLASS_NUM);
        return 0;
    }

    bool init_post_process()
    {
        if (load_label_name(LABEL_NAME_TXT_PATH, labels) < 0)
        {
            printf("Load %s failed!\n", LABEL_NAME_TXT_PATH);
            return false;
        }

        return true;
    }

    void deinit_post_process()
    {
        for (int i = 0; i < OBJ_CLASS_NUM; i++)
        {
            if (labels[i] != nullptr)
            {
                free(labels[i]);
                labels[i] = nullptr;
            }
        }
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

    // RV1106 отдает выходные тензоры в NHWC, поэтому боксы лежат по соседним адресам
    int process_i8_rv1106(int8_t* input, const int* ptr_anchor, const int grid_h, const int grid_w, const int stride,
                          const float threshold, const int32_t zp, const float scale,
                          std::vector<float>& boxes, std::vector<float>& box_scores, std::vector<int>& class_ids)
    {
        int valid_count = 0;
        int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

        const int align_c = PROP_BOX_SIZE * ANCHOR_PER_BRANCH;

        for (int h = 0; h < grid_h; h++)
        {
            for (int w = 0; w < grid_w; w++)
            {
                for (int a = 0; a < ANCHOR_PER_BRANCH; a++)
                {
                    int hw_offset = h * grid_w * align_c + w * align_c + a * PROP_BOX_SIZE;
                    int8_t* hw_ptr = input + hw_offset;
                    int8_t box_confidence = hw_ptr[4];

                    if (box_confidence >= thres_i8)
                    {
                        int8_t max_class_probs = hw_ptr[5];
                        int max_class_id = 0;
                        for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                        {
                            int8_t prob = hw_ptr[5 + k];
                            if (prob > max_class_probs)
                            {
                                max_class_id = k;
                                max_class_probs = prob;
                            }
                        }

                        float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                        float class_prob_f32 = deqnt_affine_to_f32(max_class_probs, zp, scale);
                        float limit_score = box_conf_f32 * class_prob_f32;

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
                            box_w *= (float)ptr_anchor[a * 2];
                            box_h *= (float)ptr_anchor[a * 2 + 1];

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
}

br_npu::br_npu()
{
    // Обнуляем контекст модели до инициализации
    memset(&_app_ctx, 0, sizeof(_app_ctx));
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

    // По умолчанию тип входа int8 (нормализация и квантование считаются снаружи модели).
    // Uint8 позволяет совместить нормализацию и квантование с NPU
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // По умолчанию формат NHWC: NPU 1106 в zero-copy режиме поддерживает только его
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;

    // Вход модели — тот же блок, в который RGA пишет кадр для VENC: копий нет
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
    for (uint32_t i = 0; i < io_num.n_output; i++)
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

    _app_ctx.io_num = io_num;
    _app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    _app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // Загружаем названия классов модели
    if (!init_post_process())
    {
        return false;
    }

    printf("%s: ok\n", __PRETTY_FUNCTION__);

    return true;
}

bool br_npu::inference()
{
    int ret = 0;

    // Кадр в блок пишет RGA, а читает NPU: синхронизируем кэш перед прогоном
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

    if (!post_process())
    {
        return false;
    }

    return true;
}

bool br_npu::post_process()
{
    std::vector<float> filter_boxes;
    std::vector<float> obj_probs;
    std::vector<int> class_ids;
    int valid_count = 0;

    memset(&_detections, 0, sizeof(object_detect_result_list));

    for (int i = 0; i < ANCHOR_BRANCH_NUM; i++)
    {
        // RV1106 отдает выходные тензоры в NHWC и только в i8
        const int grid_h = _app_ctx.output_attrs[i].dims[2];
        const int grid_w = _app_ctx.output_attrs[i].dims[1];
        const int stride = _app_ctx.model_height / grid_h;

        if (_app_ctx.is_quant)
        {
            valid_count += process_i8_rv1106(
                (int8_t *)(_app_ctx.output_mems[i]->virt_addr),
                anchor[i],
                grid_h,
                grid_w,
                stride,
                BOX_THRESH,
                _app_ctx.output_attrs[i].zp,
                _app_ctx.output_attrs[i].scale,
                filter_boxes,
                obj_probs,
                class_ids
            );
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
        nms(valid_count, filter_boxes, class_ids, index_array, c, NMS_THRESH);
    }

    int last_count = 0;
    for (int i = 0; i < valid_count; i++)
    {
        if (index_array[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }

        const int n = index_array[i];
        const float x1 = filter_boxes[n * 4 + 0];
        const float y1 = filter_boxes[n * 4 + 1];
        const float x2 = x1 + filter_boxes[n * 4 + 2];
        const float y2 = y1 + filter_boxes[n * 4 + 3];

        _detections.results[last_count].box.left   = clamp_value(x1, 0, _app_ctx.model_width);
        _detections.results[last_count].box.top    = clamp_value(y1, 0, _app_ctx.model_height);
        _detections.results[last_count].box.right  = clamp_value(x2, 0, _app_ctx.model_width);
        _detections.results[last_count].box.bottom = clamp_value(y2, 0, _app_ctx.model_height);
        _detections.results[last_count].prop = obj_probs[i];
        _detections.results[last_count].cls_id = class_ids[n];
        last_count++;
    }
    _detections.count = last_count;

    return true;
}

br_npu::object_detect_result_list* br_npu::get_detections()
{
    return &_detections;
}

const char* br_npu::get_class_name(const int cls_id)
{
    static char null_str[] = "null";

    if (cls_id < 0 || cls_id >= OBJ_CLASS_NUM || labels[cls_id] == nullptr)
    {
        return null_str;
    }

    return labels[cls_id];
}

bool br_npu::release()
{
    // Отвязываем входной тензор от блока кадра (сам блок принадлежит br_fsm)
    if (_app_ctx.input_mems[0] != nullptr)
    {
        rknn_destroy_mem(_ctx, _app_ctx.input_mems[0]);
        _app_ctx.input_mems[0] = nullptr;
    }

    for (uint32_t i = 0; i < _app_ctx.io_num.n_output; i++)
    {
        if (_app_ctx.output_mems[i] != nullptr)
        {
            rknn_destroy_mem(_ctx, _app_ctx.output_mems[i]);
            _app_ctx.output_mems[i] = nullptr;
        }
    }

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

    deinit_post_process();

    return true;
}

void br_npu::set_settings(const settings settings)
{
    _settings = settings;
}

