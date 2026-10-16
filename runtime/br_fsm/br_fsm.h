#pragma once

/*
*/

#include <cstdint>
#include <cstring>
#include <memory>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rk_mpi_vi.h"
    #include "rk_comm_video.h"
    #include "rk_aiq.h"
    #include "sample_comm.h"
    #include "rtsp_demo.h"
};

#include "br_mb_pool.h"
#include "br_vi.h"
#include "br_rga.h"
#include "br_venc.h"
#include "br_rtsp.h"
#include "br_npu.h"
#include "br_model.h"
#include "br_hook.h"

class br_fsm 
{
public:
    struct settings
    {
        // Ручек из config.ini
        uint16_t _width = 1920;
        uint16_t _height = 1080;
        uint8_t _id_camera = 0;
        std::string _path_to_iq_dir = "/etc/iqfiles";
        br_venc::codec _codec = br_venc::codec::H264;
        uint16_t _venc_gop = 30;
        // 0 — рассчитать битрейт от разрешения кадра
        uint32_t _venc_bitrate = 0;
        uint16_t _rtsp_port = 554;
        std::string _rtsp_path = "/live/0";
        // Ручек из model.ini
        // Тип модели: выбирает реализацию разбора выходов
        std::string _model_type = "yolov5";
        std::string _path_to_model = "./model/yolov5.rknn";
        std::string _path_to_labels = "./model/coco_80_labels_list.txt";
        std::string _path_to_anchors = "./model/anchors_yolov5.txt";
        // Размер входа модели: под него выделяется отдельный блок для NPU
        uint16_t _width_model = 640;
        uint16_t _height_model = 640;
        float _box_thresh = 0.25f;
        float _nms_thresh = 0.45f;
        // Максимальное количество рамок детекции на кадр
        int _max_detections = 128;
        // Ручек из hook.ini
        // Внешний скрипт: выключен — в поток идут кадры без разметки, процесса скрипта нет
        bool _hook_enabled = false;
        // Режим хука: false — конвеер ждет скрипт, true — не ждет
        bool _hook_async = true;
        std::string _path_to_hook = "./hook_user.py";
        std::string _path_python = "python3";
        uint32_t _hook_timeout_ms = 50;
        // Сколько терпим молчание скрипта до перезапуска: в асинхронном режиме он вправе
        // работать медленно, поэтому это отдельная ручка, а не бюджет кадра
        uint32_t _hook_stuck_ms = 2000;
        // Сколько ждем, пока скрипт поднимется: импорт cv2 и numpy на плате небыстрый
        uint32_t _hook_start_ms = 5000;
        // Параметры связки VI/RGA/VENC: это не ручки, а константы пайплайна
        uint8_t _mb_blk_count = 1;
        float _bytes_per_pixel_out = 3;
        float _bytes_per_pixel_in = 1.5;
        PIXEL_FORMAT_E _pixel_format_in = RK_FMT_YUV420SP;
        _Rga_SURF_FORMAT _pixel_format_in_rga = RK_FORMAT_YCbCr_420_SP;
        _Rga_SURF_FORMAT _pixel_format_out_rga = RK_FORMAT_RGB_888;
        // Кадр для внешнего скрипта: тот же RGB888, что уходит в VENC. Тогда возврат кадра скрипта
        // в поток - это копия без конвертации, а скрипт получает кадр тем же проходом RGA, что
        // и поток (YUV -> RGB), без лишней смены раскладки каналов
        _Rga_SURF_FORMAT _pixel_format_hook_rga = RK_FORMAT_RGB_888;
        PIXEL_FORMAT_E _pixel_format_out = RK_FMT_RGB888;
    };

    explicit br_fsm();
    ~br_fsm();

    void set_settings(const settings settings);
    bool init();
    bool start();
    bool release();
private:
    static constexpr uint64_t FPS_PERIOD_US = 1000000;

    // Тайминги стадий кадра, накопленные за окно измерения
    struct stage_stats
    {
        uint64_t vi = 0;
        uint64_t rga = 0;
        uint64_t npu = 0;
        uint64_t hook = 0;
        uint64_t flush = 0;
        uint64_t venc = 0;
        uint64_t rtsp = 0;
    };

    settings _settings;

    bool inited = false;

    void print_fps();
    // Отладка кадров: времена стадий текущего кадра и число детекций.
    // Печатает только под определением debug_frame (см. DEBUG_FRAME в CMakeLists)
    void print_frame_debug(const stage_stats& frame_us);
    uint64_t get_current_time_us() const;
    uint64_t take_stage_time();

    uint32_t _fps_frames = 0;
    uint64_t _fps_start_time_us = 0;
    uint64_t _stage_time_us = 0;
    uint32_t _inference_count_prev = 0;
    stage_stats _stage_us;

    br_mb_pool _mb_pool;
    br_mb_pool _mb_pool_npu;
    // Пул под кадр внешнего скрипта: формат потока (RGB888), разрешение потока
    br_mb_pool _mb_pool_hook;
    br_vi _vi;
    br_rga _brga;
    br_venc _venc;
    br_rtsp _rtsp;
    // Разбор выходов модели: создается по типу из конфига, br_npu только вызывает его методы
    std::unique_ptr<br_model> _model;
    br_npu _npu;
    // Внешний скрипт пользователя: рисует поверх кадра, который уходит в поток
    br_hook _hook;
};