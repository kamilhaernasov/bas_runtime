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
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "im2d_buffer.h"

class rga
{
public:
    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        PIXEL_FORMAT_E _pixel_format_out;
        PIXEL_FORMAT_E _pixel_format_in;
        _Rga_SURF_FORMAT _pixel_format_in_rga;
        _Rga_SURF_FORMAT _pixel_format_out_rga;
        float _pixels_per_byte_out;
        float _pixels_per_byte_in;
    };

    explicit rga();
    ~rga();

    bool init(MB_BLK ptr_blk);
    void set_settings(const settings settings);

    bool process_frame(VIDEO_FRAME_INFO_S* ptr_frame);

    bool release();
private:
    static constexpr uint8_t ROUND = 15;

    settings _settings;

    uint32_t _size_for_venc;
    uint32_t _size_from_vi;

    rga_buffer_handle_t _handle_buffer_venc = 0;
    rga_buffer_t _buffer_for_venc {};
};