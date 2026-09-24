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
        PIXEL_FORMAT_E _pixel_format_for_venc;
    };

    explicit rga();
    ~rga();

    bool init();
    void set_settings(const settings settings);
    bool create_buffer(MB_BLK ptr_blk);

    int process_frame(VIDEO_FRAME_INFO_S* ptr_frame);

    uint32_t get_wstride() const;
    uint32_t get_bgr_size() const;
private:
    settings _settings;

    uint32_t _wstride;
    uint32_t _bgr_size;

    int dst_fd = 0;

    rga_buffer_handle_t _src_handle = 0;
    rga_buffer_handle_t _dst_handle = 0;
    rga_buffer_t        _src_rga{};
    rga_buffer_t        _dst_rga{};
};