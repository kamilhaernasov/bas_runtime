#pragma once

#include <cstring>
#include <string>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rk_mpi_vi.h"
    #include "rk_comm_video.h"
    #include "rk_aiq.h"
    #include "sample_comm.h"
    #include "rtsp_demo.h"
    #include "yolov5.h"
};

class br_npu
{
public:
    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        std::string _model_path;
    };

    explicit br_npu();
    ~br_npu();

    bool init();
    void set_settings(const settings settings);
private:
    settings _settings;

    rknn_context _ctx;
    rknn_app_context_t _app_ctx;
};