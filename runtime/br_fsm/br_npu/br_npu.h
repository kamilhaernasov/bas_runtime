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
 * с сохранением пропорций, поля заливаются цветом, с которым обучалась модель. Буфер под
 * вход модели и сам проход RGA — в br_rga, модуль его только просит о подготовке кадра.
 * Прогон модели и разбор ее выходов идут в отдельном потоке, поэтому цикл обработки кадров
 * не ждет NPU: если вход занят прогоном, кадр для детекции просто пропускается.
 * 
 */

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rk_comm_video.h"
    #include "rknn_api.h"
};

#include "br_rga.h"

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

    // ptr_blk — блок под вход модели, ptr_rga — модуль RGA, который в него пишет кадр
    bool init(MB_BLK ptr_blk, br_rga* ptr_rga);
    bool release();
    void set_settings(const settings settings);

    // Заполняет вход модели кадром. Не блокирует: если NPU занят прогоном предыдущего
    // кадра, вызов сразу возвращается, а кадр пропускается
    bool prepare_input(VIDEO_FRAME_INFO_S* ptr_frame);

    object_detect_result_list get_detections();
    uint32_t get_inference_count();
    static const char* get_class_name(const int cls_id);
private:
    static constexpr float BOX_THRESH = 0.25f;
    static constexpr float NMS_THRESH = 0.45f;

    // Состояние буфера входа модели
    enum class input_state
    {
        free,   // свободен, цикл может заполнять
        ready,  // заполнен, ждет прогона
        busy    // идет прогон и разбор
    };

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

    bool run();
    bool process_outputs();
    bool post_process(object_detect_result_list& detections);
    bool map_detections_to_frame(object_detect_result_list& detections);

    void worker();

    settings _settings;

    rknn_context _ctx;
    rknn_app_context_t _app_ctx;

    // RGA-модуль, который готовит кадр под вход модели (принадлежит br_fsm)
    br_rga* _ptr_rga = nullptr;

    // Поток прогона модели и его связь с циклом
    std::thread _worker;
    std::mutex _mtx;
    std::condition_variable _cv;
    input_state _input_state = input_state::free;
    bool _stop = false;

    // Результаты детекции в координатах кадра и их защита
    std::mutex _mtx_detections;
    object_detect_result_list _detections {};
    uint32_t _inference_count = 0;
};