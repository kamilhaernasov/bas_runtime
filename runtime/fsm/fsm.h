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

class fsm 
{
public:
    explicit fsm();
    ~fsm();

    bool init();

    void set_width(const uint16_t width);
    void set_height(const uint16_t height);
    uint16_t get_width() const;
    uint16_t get_height() const;

    bool create_mb_pool(const uint8_t mb_cnt);
    bool build_h264_frame(MB_BLK* mb_blk);
    bool isp_init();
    bool mpi_init();
    bool rtsp_init();
    bool vi_dev_init();
    bool vi_channel_init();
    bool venc_init();

private:
    uint16_t _width = 1920;
    uint16_t _height = 1080;

    MB_POOL_CONFIG_S _mb_pool_cfg;
    MB_POOL _mb_pool;

    VIDEO_FRAME_INFO_S _h264_frame;

    rtsp_demo_handle _rtsp_handle;
    rtsp_session_handle _rtsp_session;

    VI_DEV_ATTR_S _vi_dev_attr;
	VI_DEV_BIND_PIPE_S _vi_dev_bind_pipe;
    VI_CHN_ATTR_S _vi_chn_attr;

    VENC_RECV_PIC_PARAM_S _venc_recv_param;
	VENC_CHN_ATTR_S _venc_chn_attr;
};