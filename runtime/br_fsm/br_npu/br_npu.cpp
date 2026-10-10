#include "br_npu.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <set>

// Якоря yolov5 (три масштаба), как в примере
static const int anchors[3][6] = {{10, 13, 16, 30, 33, 23},
                                  {30, 61, 62, 45, 59, 119},
                                  {116, 90, 156, 198, 373, 326}};

static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

static int32_t clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);

    return (int32_t)f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;

    return (int8_t)clip(dst_val, -128, 127);
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;

    return u <= 0.f ? 0.f : (i / u);
}

// Подавление немаксимумов в пределах одного класса
static int nms(int valid_count, std::vector<float>& output_locations, std::vector<int> class_ids,
               std::vector<int>& order, int filter_id, float threshold)
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

// Сортировка по убыванию вероятности с одновременной перестановкой индексов
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
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

// Декодирование выходов yolov5. На 1106 раскладка выходов своя (HWNC) и только int8
static int process_i8_rv1106(int8_t* input, const int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                             std::vector<float>& boxes, std::vector<float>& box_scores, std::vector<int>& class_id,
                             float threshold, int32_t zp, float scale)
{
    int valid_count = 0;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    int anchor_per_branch = 3;
    int align_c = PROP_BOX_SIZE * anchor_per_branch;

    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < anchor_per_branch; a++)
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
                        box_w *= (float)anchor[a * 2];
                        box_h *= (float)anchor[a * 2 + 1];

                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                        box_scores.push_back(limit_score);
                        class_id.push_back(max_class_id);
                        valid_count++;
                    }
                }
            }
        }
    }

    return valid_count;
}

br_npu::br_npu()
{

}

br_npu::~br_npu()
{

}

bool br_npu::init()
{
    int ret = rknn_init(&_ctx, (char*)_settings._model_path.c_str(), 0, 0, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_init fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("%s: rknn_query fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < (int)io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query input_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < (int)io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        // В zero-copy режиме запрашиваем нативные атрибуты выхода
        ret = rknn_query(_ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("%s: rknn_query output_attrs fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
    }

    // По умолчанию тип входа int8 (нормализация и квантование считаются снаружи).
    // Поставив uint8, отдаем нормализацию и квантование в npu
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // 1106 в zero-copy режиме умеет только NHWC
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;

    _app_ctx.input_mems[0] = rknn_create_mem(_ctx, input_attrs[0].size_with_stride);
    ret = rknn_set_io_mem(_ctx, _app_ctx.input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("%s: input_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        _app_ctx.output_mems[i] = rknn_create_mem(_ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(_ctx, _app_ctx.output_mems[i], &output_attrs[i]);
        if (ret < 0)
        {
            printf("%s: output_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
            return false;
        }
    }

    _app_ctx.rknn_ctx = _ctx;
    _app_ctx.is_quant = output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;

    _app_ctx.io_num = io_num;
    _app_ctx.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    _app_ctx.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

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

    printf("model input height=%d, width=%d, channel=%d\n",
           _app_ctx.model_height, _app_ctx.model_width, _app_ctx.model_channel);

    _model_width = _app_ctx.model_width;
    _model_height = _app_ctx.model_height;

    load_labels();

    return true;
}

void br_npu::set_settings(const settings settings)
{
    _settings = settings;
}

bool br_npu::inference(const uint8_t* ptr_model_input, object_detect_result_list* ptr_od_results)
{
    if (ptr_model_input == nullptr)
    {
        printf("%s: ptr_model_input is null!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Кладем кадр в память входа npu
    memcpy(_app_ctx.input_mems[0]->virt_addr, ptr_model_input, _model_width * _model_height * 3);

    int ret = rknn_run(_ctx, nullptr);
    if (ret < 0)
    {
        printf("%s: rknn_run fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    post_process(ptr_od_results);

    return true;
}

bool br_npu::release()
{
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

    for (int i = 0; i < (int)_app_ctx.io_num.n_input; i++)
    {
        if (_app_ctx.input_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx.rknn_ctx, _app_ctx.input_mems[i]);
            _app_ctx.input_mems[i] = nullptr;
        }
    }
    for (int i = 0; i < (int)_app_ctx.io_num.n_output; i++)
    {
        if (_app_ctx.output_mems[i] != nullptr)
        {
            rknn_destroy_mem(_app_ctx.rknn_ctx, _app_ctx.output_mems[i]);
            _app_ctx.output_mems[i] = nullptr;
        }
    }

    if (_ctx != 0)
    {
        rknn_destroy(_ctx);
        _ctx = 0;
        _app_ctx.rknn_ctx = 0;
    }

    return true;
}

void br_npu::load_labels()
{
    std::ifstream file(_settings._label_path);
    if (!file.is_open())
    {
        printf("%s: open %s fail!\n", __PRETTY_FUNCTION__, _settings._label_path.c_str());
        return;
    }

    std::string line;
    while ((int)_labels.size() < OBJ_CLASS_NUM && std::getline(file, line))
    {
        _labels.push_back(line);
    }
}

void br_npu::post_process(object_detect_result_list* ptr_od_results)
{
    std::vector<float> filter_boxes;
    std::vector<float> obj_probs;
    std::vector<int> class_id;
    int valid_count = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = _app_ctx.model_width;
    int model_in_h = _app_ctx.model_height;

    memset(ptr_od_results, 0, sizeof(object_detect_result_list));

    for (int i = 0; i < 3; i++)
    {
        // На 1106 выходы в NHWC, поэтому высота и ширина стоят в другом порядке
        grid_h = _app_ctx.output_attrs[i].dims[2];
        grid_w = _app_ctx.output_attrs[i].dims[1];
        stride = model_in_h / grid_h;
        // 1106 поддерживает только int8
        if (_app_ctx.is_quant)
        {
            valid_count += process_i8_rv1106(
                (int8_t*)(_app_ctx.output_mems[i]->virt_addr),
                anchors[i],
                grid_h, grid_w, model_in_h, model_in_w, stride,
                filter_boxes, obj_probs, class_id, BOX_THRESH,
                _app_ctx.output_attrs[i].zp, _app_ctx.output_attrs[i].scale
            );
        }
    }

    // Ничего не нашли
    if (valid_count <= 0)
    {
        return;
    }

    std::vector<int> index_array;
    for (int i = 0; i < valid_count; ++i)
    {
        index_array.push_back(i);
    }
    quick_sort_indice_inverse(obj_probs, 0, valid_count - 1, index_array);

    std::set<int> class_set(std::begin(class_id), std::end(class_id));
    for (auto c : class_set)
    {
        nms(valid_count, filter_boxes, class_id, index_array, c, NMS_THRESH);
    }

    int last_count = 0;
    ptr_od_results->count = 0;

    for (int i = 0; i < valid_count; ++i)
    {
        if (index_array[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = index_array[i];

        float x1 = filter_boxes[n * 4 + 0];
        float y1 = filter_boxes[n * 4 + 1];
        float x2 = x1 + filter_boxes[n * 4 + 2];
        float y2 = y1 + filter_boxes[n * 4 + 3];
        int id = class_id[n];
        float obj_conf = obj_probs[i];

        ptr_od_results->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w));
        ptr_od_results->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h));
        ptr_od_results->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w));
        ptr_od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h));
        ptr_od_results->results[last_count].prop = obj_conf;
        ptr_od_results->results[last_count].cls_id = id;
        last_count++;
    }

    ptr_od_results->count = last_count;
}

int br_npu::get_model_width() const
{
    return _model_width;
}

int br_npu::get_model_height() const
{
    return _model_height;
}

const char* br_npu::class_name(int cls_id) const
{
    if (cls_id < 0 || cls_id >= (int)_labels.size())
    {
        return "null";
    }

    return _labels[cls_id].c_str();
}
