#pragma once

#include <cstdint>
#include <vector>

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

class venc
{
public:
    enum class codec
    {
        H264 = 0
    };

    struct settings
    {
        codec _codec;
        uint32_t _bitrate;
        uint16_t _gop;
        uint16_t _width;
        uint16_t _height;
        uint8_t _bytes_per_pixels;
        rkPIXEL_FORMAT_E _pixel_format;
    };

    explicit venc();
    ~venc();

    void set_settings(const settings settings);

    bool init(const uint16_t channel, MB_BLK ptr_blk);
    bool release();

    bool prepare_frame(VIDEO_FRAME_INFO_S* frame);
    bool process_frame();
    
    VENC_STREAM_S* get_codec_frame();
    bool release_codec_frame();
private:
    uint64_t get_current_time_us() const;

    settings _settings;

    uint16_t _channel;

    VENC_STREAM_S _codec_frame;	
    VIDEO_FRAME_INFO_S _frame;
};

