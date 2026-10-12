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
#include <vector>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rknn_api.h"
};

class br_npu
{
public:
    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        std::string _model_path;
        std::string _path_to_labels;
        std::string _path_to_anchors;
        float _box_thresh;
        float _nms_thresh;
        // Сколько рамок детекции помещается в кадр
        int _max_detections;
        // Геометрия вписывания кадра во вход модели. Считает ее RGA: он и укладывает
        // кадр во вход, а NPU по ней возвращает боксы в координаты кадра
        float _scale;
        uint16_t _pad_x;
        uint16_t _pad_y;
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
        // Размер списка задает настройка _max_detections
        std::vector<object_detect_result> results;
    };

    explicit br_npu();
    ~br_npu();

    // ptr_blk — блок под вход модели: в него кадр укладывает RGA, его же читает NPU
    bool init(MB_BLK ptr_blk);
    bool release();
    void set_settings(const settings settings);

    // Свободен ли вход модели. Пока идет прогон или разбор предыдущего кадра, кадр
    // для детекции пропускается: RGA в него не пишет, работа не тратится
    bool is_input_free();
    // Кадр уложен во вход — отдаем его в поток прогона
    void submit_input();

    object_detect_result_list get_detections();
    uint32_t get_inference_count();
    static const char* get_class_name(const int cls_id);
private:
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
        // Входных и выходных тензоров столько, сколько их у модели: числа узнаются из
        // атрибутов, поэтому здесь не массивы фиксированной длины
        std::vector<rknn_tensor_mem*> input_mems;
        std::vector<rknn_tensor_mem*> output_mems;
        rknn_dma_buf img_dma_buf;
        int model_channel;
        int model_width;
        int model_height;
        // Форма голов модели: все три числа выводятся в init из самой модели и файлов
        // рядом с ней, потому что описывают именно ее
        int anchor_branch_num;
        int anchor_per_branch;
        int prop_box_size;
        int class_num;
        bool is_quant;
    };

    static void dump_tensor_attr(const rknn_tensor_attr* ptr_attr);

    bool run();
    bool process_outputs();
    bool post_process(object_detect_result_list& detections);
    // Разбор одной головы модели: раскладка NHWC, деквантование и декодирование боксов
    int process_head(const int head, std::vector<float>& boxes, std::vector<float>& box_scores,
                     std::vector<int>& class_ids);
    bool map_detections_to_frame(object_detect_result_list& detections);

    void worker();

    settings _settings;

    // Контекст модели обнуляется инициализатором: memset по нему нельзя, внутри вектор
    rknn_context _ctx = 0;
    rknn_app_context_t _app_ctx {};

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