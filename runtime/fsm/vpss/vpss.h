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
    #include "rk_mpi_vpss.h"
};

class vpss
{
public:
    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        PIXEL_FORMAT_E _format_input;
        PIXEL_FORMAT_E _format_output;
    };

    explicit vpss();
    ~vpss();

    bool init();
    void set_settings(const settings settings);
    MPP_CHN_S* get_chn_bind();
    bool receive_frame();
    VIDEO_FRAME_INFO_S* get_frame();
    bool release_frame();
private:
    settings _settings;
    MPP_CHN_S _mpp_chn;

    VIDEO_FRAME_INFO_S _frame;
};