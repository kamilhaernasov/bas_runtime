#pragma once

/**
 * @file fsm.h
 * @author khaernasov-ka
 * @brief Стейт-машина рантайма.
 * @version 0.1
 * @date 2026-09-23
 * 
 * Немного об аппаратных блоках внутри этого камня RV1106G3:
 * ├── VI - The VI module captures video images, cuts and zooms them, and outputs multi-channel image data with different resolutions.
 *      ├── VICAP 
 *      ├── ISP 
 *      ├── ISPP 
 *      ├── AIQ 
 * ├── VPSS - The VPSS module receives the image transmitted by the VI module, and can process the image by cutting, scaling, rotating, pixel format conversion and the like.
 * ├── VENC - The VENC module can directly receive the image captured by the VI module or the image data output after VPSS processing, superimpose the OSD image set by the user through the RGN module, and then encode and output the corresponding code stream according to different protocols.
 * 
 */

#include <cstdint>
#include <cstring>

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

#include "mb_pool.h"
#include "vi.h"
#include "vpss.h"
#include "venc.h"
#include "rtsp.h"

class fsm 
{
public:
    struct settings
    {
        uint16_t _width = 640;
        uint16_t _height = 640;
        uint8_t _mb_blk_count = 1;
        uint8_t _bytes_per_pixel = 3;
        uint8_t _id_camera = 0;
        std::string _path_to_iq_dir = "/etc/iqfiles";
        PIXEL_FORMAT_E _pixel_format = RK_FMT_YUV420SP;
        venc::codec _codec = venc::codec::H264;
        uint8_t _venc_gop = 1;
    };

    explicit fsm();
    ~fsm();

    bool init();
    bool start();
    bool release();
private:
    settings _settings;

    mb_pool _mb_pool;
    vi _vi;
    vpss _vpss;
    venc _venc;
    rtsp _rtsp;
};