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

class rtsp
{
public:
    struct settings
    {
        uint16_t _port;
        std::string _path;
        rtsp_codec_id _codec_rtsp;
    };

    explicit rtsp();
    ~rtsp();

    void init();
    void release();
    
    void set_settings(const settings settings);

    void send_frame(uint8_t* ptr_data, const uint32_t len_data, const uint64_t pts_data);
private:
    settings _settings;

    rtsp_demo_handle _rtsp_handle;
    rtsp_session_handle _rtsp_session;
};