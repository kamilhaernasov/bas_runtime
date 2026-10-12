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
 * Вход модели — тот же кадр, который RGA готовит для VENC (RGB888 нужного размера):
 * модуль привязывает этот блок к входному тензору, поэтому ни копий, ни лишних
 * конвертаций нет. Выходные тензоры квантованы, поэтому после прогона выполняется
 * деквантование, разбор боксов и NMS. Координаты боксов — уже в пикселях кадра.
 * 
 */

#include <cstdint>
#include <cstring>
#include <string>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rknn_api.h"
};

class br_npu
{
public:
    // Максимальное количество детекций на кадр
    static constexpr int OBJ_NUMB_MAX_SIZE = 128;

    struct settings
    {
        std::string _model_path;
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

    bool init(MB_BLK ptr_blk);
    bool release();
    void set_settings(const settings settings);

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

    bool post_process();

    settings _settings;

    rknn_context _ctx;
    rknn_app_context_t _app_ctx;

    // Результаты детекции последнего кадра
    object_detect_result_list _detections {};
};