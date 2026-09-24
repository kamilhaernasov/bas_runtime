#include "fsm.h"

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

fsm::fsm()
{

}

fsm::~fsm()
{

}

bool fsm::init()
{
    // MB_POOL
    _mb_pool.init(_settings._mb_blk_count, _settings._width, _settings._height, _settings._bytes_per_pixel);
    _mb_pool.create_mb_blk(RK_TRUE);

    // VI
    vi::settings vi_settings 
    {
        ._path_to_iq_dir =  _settings._path_to_iq_dir,
        ._id_camera = _settings._id_camera,
        ._pixel_format = _settings._pixel_format,
        ._width = _settings._width,
        ._height = _settings._height,
    };
    _vi.set_settings(vi_settings);
    _vi.init();

    // VPSS
    vpss::settings vpss_setiings
    {
        ._width = _settings._width,
        ._height = _settings._height,
        ._format_input = RK_FMT_YUV420SP,
        ._format_output = RK_FMT_RGB888,
    };
    _vpss.set_settings(vpss_setiings);
    _vpss.init();

    RK_MPI_SYS_Bind(_vi.get_chn_bind(), _vpss.get_chn_bind());

    // VENC
    venc::settings venc_settings
    {
        ._codec = _settings._codec,
        ._bitrate = _settings._width * _settings._height / 8 * 30,
        ._gop = _settings._venc_gop,
        ._width = _settings._width,
        ._height = _settings._height,
        ._bytes_per_pixels = _settings._bytes_per_pixel,
        ._pixel_format = RK_FMT_RGB888,
    };
    _venc.set_settings(venc_settings);
    _venc.init(0, _mb_pool.get_mb_blk(0));

    // RTSP
    rtsp::settings rtsp_settings
    {
        ._port = 554,
        ._path = "/live/0",
        ._codec_rtsp = RTSP_CODEC_ID_VIDEO_H264,
    };
    _rtsp.set_settings(rtsp_settings);
    _rtsp.init();
}

bool fsm::start()
{
    while(1)
    {
        _vpss.receive_frame();

        _venc.exec_frame_from_vi(_vpss.get_frame());
        _venc.exec_frame_to_codec();

        _rtsp.send_frame(
            reinterpret_cast<uint8_t*>(_mb_pool.get_handle_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        );
        _vpss.release_frame();
        _venc.release_frame();
    }

    return true;
}

 bool fsm::release()
 {
    _mb_pool.release();
    _vi.release();
    RK_MPI_VPSS_StopGrp(0);
    RK_MPI_VPSS_DestroyGrp(0);
    _venc.release();
    _rtsp.release();
    RK_MPI_SYS_Exit();

    return true;
 }