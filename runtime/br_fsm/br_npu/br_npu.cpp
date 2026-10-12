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
    // Цвет полей кадра входа модели (114, 114, 114) — такой же, как при обучении модели
    constexpr int PADDING_COLOR = 0x727272;

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

bool br_npu::init()
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

    // Выделяем буфер входа: в него пишет RGA, из него читает NPU
    if (!init_input_buffer(&input_attrs[0]))
    {
        return false;
    }

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

bool br_npu::init_input_buffer(rknn_tensor_attr* ptr_input_attr)
{
    // Считаем параметры вписывания кадра во вход модели: масштаб с сохранением
    // пропорций и поля по краям кадра входа
    const float scale_x = (float)_app_ctx.model_width / (float)_settings._width;
    const float scale_y = (float)_app_ctx.model_height / (float)_settings._height;
    _scale = scale_x < scale_y ? scale_x : scale_y;
    _content_w = (uint16_t)((float)_settings._width * _scale);
    _content_h = (uint16_t)((float)_settings._height * _scale);
    _pad_x = (uint16_t)((_app_ctx.model_width - _content_w) / 2);
    _pad_y = (uint16_t)((_app_ctx.model_height - _content_h) / 2);
    printf("%s: letterbox scale=%f, content=%dx%d, pad=(%u, %u)\n",
           __PRETTY_FUNCTION__, _scale, _content_w, _content_h, _pad_x, _pad_y);

    // Буфер входа выделяем сами: в него пишет RGA, из него читает NPU
    MB_POOL_CONFIG_S mb_pool_cfg;
    memset(&mb_pool_cfg, 0, sizeof(MB_POOL_CONFIG_S));
    mb_pool_cfg.u64MBSize = ptr_input_attr->size_with_stride;
    mb_pool_cfg.u32MBCnt = 1;
    mb_pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;
    mb_pool_cfg.bPreAlloc = RK_TRUE;
    _mb_pool = RK_MPI_MB_CreatePool(&mb_pool_cfg);
    if (_mb_pool == MB_INVALID_POOLID)
    {
        printf("%s: RK_MPI_MB_CreatePool fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    _mb_blk = RK_MPI_MB_GetMB(_mb_pool, mb_pool_cfg.u64MBSize, RK_TRUE);
    if (_mb_blk == MB_INVALID_HANDLE)
    {
        printf("%s: RK_MPI_MB_GetMB fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Привязываем блок памяти к входному тензору модели
    _app_ctx.input_mems[0] = rknn_create_mem_from_mb_blk(_ctx, _mb_blk, 0);
    if (_app_ctx.input_mems[0] == nullptr)
    {
        printf("%s: rknn_create_mem_from_mb_blk fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    int ret = rknn_set_io_mem(_ctx, _app_ctx.input_mems[0], ptr_input_attr);
    if (ret < 0)
    {
        printf("%s: input_mems rknn_set_io_mem fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    // Прокидываем блок памяти в RGA и получаем буфер под кадр входа модели
    _handle_in = importbuffer_fd(RK_MPI_MB_Handle2Fd(_mb_blk), mb_pool_cfg.u64MBSize);
    if (_handle_in == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    _buffer_in = wrapbuffer_handle(_handle_in, _app_ctx.model_width, _app_ctx.model_height, RK_FORMAT_RGB_888);
    _buffer_in.wstride = (ptr_input_attr->w_stride > 0) ? (int)ptr_input_attr->w_stride : _app_ctx.model_width;

    printf("%s: input w_stride=%u, size_with_stride=%u\n",
           __PRETTY_FUNCTION__, ptr_input_attr->w_stride, ptr_input_attr->size_with_stride);

    return true;
}

bool br_npu::prepare_input(VIDEO_FRAME_INFO_S* ptr_frame)
{
    int ret = 0;

    // Прокидываем блок памяти кадра с VI в RGA
    rga_buffer_handle_t handle_from_vi = importbuffer_fd(
        RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk),
        _settings._width * _settings._height * 3 / 2
    );
    if (handle_from_vi == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    rga_buffer_t buffer_from_vi = wrapbuffer_handle(
        handle_from_vi,
        _settings._width,
        _settings._height,
        _settings._pixel_format_in_rga
    );

    // Заливаем кадр входа цветом полей, чтобы поля не собирались из прошлых кадров
    im_rect rect_input { 0, 0, _app_ctx.model_width, _app_ctx.model_height };
    ret = imfill(_buffer_in, rect_input, PADDING_COLOR);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: imfill fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        return false;
    }

    // Масштабируем кадр под вход модели, конвертируем цвет и кладем его по центру
    im_rect rect_from_vi { 0, 0, _settings._width, _settings._height };
    im_rect rect_content { _pad_x, _pad_y, _content_w, _content_h };
    ret = improcess(buffer_from_vi, _buffer_in, {}, rect_from_vi, rect_content, {}, -1, nullptr, nullptr, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: improcess fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        return false;
    }

    ret = releasebuffer_handle(handle_from_vi);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s, ret: %d\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret), ret);
        return false;
    }

    // Отдаем кадр входа устройству
    ret = rknn_mem_sync(_ctx, _app_ctx.input_mems[0], RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0)
    {
        printf("%s: rknn_mem_sync fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

bool br_npu::inference()
{
    int ret = 0;

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

    if (!map_detections_to_frame())
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

bool br_npu::map_detections_to_frame()
{
    for (int i = 0; i < _detections.count; i++)
    {
        object_detect_result* ptr_result = &(_detections.results[i]);

        // Переводим боксы из пространства входа модели в координаты кадра
        ptr_result->box.left   = clamp_value(((float)ptr_result->box.left   - _pad_x) / _scale, 0, _settings._width - 1);
        ptr_result->box.top    = clamp_value(((float)ptr_result->box.top    - _pad_y) / _scale, 0, _settings._height - 1);
        ptr_result->box.right  = clamp_value(((float)ptr_result->box.right  - _pad_x) / _scale, 0, _settings._width - 1);
        ptr_result->box.bottom = clamp_value(((float)ptr_result->box.bottom - _pad_y) / _scale, 0, _settings._height - 1);
    }

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
    int ret = 0;

    // Освобождаем буфер входа
    if (_handle_in != 0)
    {
        ret = releasebuffer_handle(_handle_in);
        if (ret != IM_STATUS_SUCCESS)
        {
            printf("%s: releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
            return false;
        }
        _handle_in = 0;
    }

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

    if (_mb_blk != MB_INVALID_HANDLE)
    {
        RK_MPI_MB_ReleaseMB(_mb_blk);
        _mb_blk = MB_INVALID_HANDLE;
    }

    if (_mb_pool != MB_INVALID_POOLID)
    {
        RK_MPI_MB_DestroyPool(_mb_pool);
        _mb_pool = MB_INVALID_POOLID;
    }

    deinit_post_process();

    return true;
}

void br_npu::set_settings(const settings settings)
{
    _settings = settings;
}

