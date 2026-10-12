#pragma once

/**
 * @file br_npu.h
 * @author khaernasov-ka
 * @brief Модуль инференса на NPU.
 * @version 0.1
 * @date 2026-10-12
 * 
 * NPU на этом камне работает через RKNN API:
 * ├── rknn_init       - загрузка модели .rknn и создание контекста
 * ├── rknn_query      - запрос атрибутов входных и выходных тензоров модели
 * ├── rknn_create_mem - выделение памяти под тензор
 * ├── rknn_set_io_mem - привязка памяти тензора (zero-copy: без промежуточных копий)
 * └── rknn_run        - прогон модели
 * 
 * Кадр с VI (YUV420SP) упаковывается под вход модели на RGA: кадр вписывается во вход
 * с сохранением пропорций, поля заливаются цветом, с которым обучалась модель.
 * Выходные тензоры квантованы, поэтому после прогона выполняется деквантование,
 * разбор боксов и NMS, а координаты пересчитываются в координаты кадра.
 * 
 */

#include <cstdint>
#include <cstring>
#include <string>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rk_comm_video.h"
    #include "rknn_api.h"
};

#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "im2d_buffer.h"

class br_npu
{
public:
    // Максимальное количество детекций на кадр
    static constexpr int OBJ_NUMB_MAX_SIZE = 128;

    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        std::string _model_path;
        _Rga_SURF_FORMAT _pixel_format_in_rga;
    };

    struct image_rect_t
    {
        int left;
        int top;
        int right;
        int bottom;
    };

    struct object_detect_result
    {
        image_rect_t box;
        float prop;
        int cls_id;
    };

    struct object_detect_result_list
    {
        int id;
        int count;
        object_detect_result results[OBJ_NUMB_MAX_SIZE];
    };

    explicit br_npu();
    ~br_npu();

    bool init();
    bool release();
    void set_settings(const settings settings);

    bool prepare_input(VIDEO_FRAME_INFO_S* ptr_frame);
    bool inference();

    object_detect_result_list* get_detections();
    static const char* get_class_name(const int cls_id);
private:
    static constexpr float BOX_THRESH = 0.25f;
    static constexpr float NMS_THRESH = 0.45f;

    // Буфер DMA, которым NPU обменивается с внешней памятью
    struct rknn_dma_buf
    {
        char* dma_buf_virt_addr;
        int dma_buf_fd;
        int size;
    };

    // Контекст модели. Повторяет структуру из примера
    // luckfox_pico_rtsp_yolov5, но является собственной: SDK-шный yolov5.h не тянем.
    struct rknn_app_context_t
    {
        rknn_context rknn_ctx;
        rknn_input_output_num io_num;
        rknn_tensor_attr* input_attrs;
        rknn_tensor_attr* output_attrs;
        rknn_tensor_mem* input_mems[1];
        rknn_tensor_mem* output_mems[3];
        rknn_dma_buf img_dma_buf;
        int model_channel;
        int model_width;
        int model_height;
        bool is_quant;
    };

    static void dump_tensor_attr(const rknn_tensor_attr* ptr_attr);

    bool init_input_buffer(rknn_tensor_attr* ptr_input_attr);
    bool post_process();
    bool map_detections_to_frame();

    settings _settings;

    rknn_context _ctx;
    rknn_app_context_t _app_ctx;

    // Параметры вписывания кадра во вход модели
    float _scale = 0;
    uint16_t _content_w = 0;
    uint16_t _content_h = 0;
    uint16_t _pad_x = 0;
    uint16_t _pad_y = 0;

    // Буфер входа модели: пишет RGA, читает NPU
    MB_POOL _mb_pool = MB_INVALID_POOLID;
    MB_BLK _mb_blk = MB_INVALID_HANDLE;
    rga_buffer_handle_t _handle_in = 0;
    rga_buffer_t _buffer_in {};

    // Результаты детекции в координатах кадра
    object_detect_result_list _detections {};
};