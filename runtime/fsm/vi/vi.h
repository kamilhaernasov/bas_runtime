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
};

class vi
{
public:
    struct settings
    {
        std::string _path_to_iq_dir;
        uint8_t _id_camera;
        PIXEL_FORMAT_E _pixel_format;
        uint16_t _width;
        uint16_t _height;
    };

    explicit vi();
    ~vi();

    void set_settings(const settings settings);
    void init();
    void release();

    VIDEO_FRAME_INFO_S* get_frame();
    bool receive_frame_from_channel();

    void release_frame();
private:
    settings _settings;

    VIDEO_FRAME_INFO_S _frame;
};