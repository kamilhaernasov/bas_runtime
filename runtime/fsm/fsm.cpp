#include "fsm.h"

#include <iostream>

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

    // VENC
    venc::settings venc_settings
    {
        ._codec = _settings._codec,
        ._bitrate = _settings._width * _settings._height / 8 * 30,
        ._gop = _settings._venc_gop,
        ._width = _settings._width,
        ._height = _settings._height,
        ._bytes_per_pixels = _settings._bytes_per_pixel,
        ._pixel_format = _settings._pixel_format,
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

    while(1)
    {
        _vi.exec_frame();
        _venc.exec_frame_from_vi(_vi.get_frame());
        _venc.exec_frame_to_codec();

        _rtsp.send_frame
        (
            reinterpret_cast<uint8_t*>(_mb_pool.get_handle_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        );
        
        _vi.release_frame();
        _venc.release_frame();
    }

    _mb_pool.release();
    _vi.release();
    _venc.release();
    _rtsp.release();
    RK_MPI_SYS_Exit();

    return true;
}