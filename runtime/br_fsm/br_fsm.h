#pragma once

/**
 * @file br_fsm.h
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

#include "br_mb_pool.h"
#include "br_vi.h"
#include "br_rga.h"
#include "br_venc.h"
#include "br_rtsp.h"
#include "br_npu.h"

class br_fsm 
{
public:
    struct settings
    {
        uint16_t _width = 1920;
        uint16_t _height = 1080;
        uint8_t _mb_blk_count = 1;
        float _bytes_per_pixel_out = 3;
        float _bytes_per_pixel_in = 1.5;
        uint8_t _id_camera = 0;
        std::string _path_to_iq_dir = "/etc/iqfiles";
        std::string _path_to_model = "./model/yolov5.rknn";
        PIXEL_FORMAT_E _pixel_format_in = RK_FMT_YUV420SP;
        _Rga_SURF_FORMAT _pixel_format_in_rga = RK_FORMAT_YCbCr_420_SP;
        _Rga_SURF_FORMAT _pixel_format_out_rga = RK_FORMAT_RGB_888;
        PIXEL_FORMAT_E _pixel_format_out = RK_FMT_RGB888;
        br_venc::codec _codec = br_venc::codec::H264;
        uint8_t _venc_gop = 1;
    };

    explicit br_fsm();
    ~br_fsm();

    bool init();
    bool start();
    bool release();
private:
    settings _settings;

    bool inited = false;

    bool draw_detections(void* ptr_frame);

    br_mb_pool _mb_pool;
    br_vi _vi;
    br_rga _brga;
    br_venc _venc;
    br_rtsp _rtsp;
    br_npu _npu;
};