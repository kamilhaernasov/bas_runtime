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
    _mb_pool.init(_settings._mb_blk_count, _settings._width * _settings._height * _settings._bytes_per_pixel_out);
    _mb_pool.create_mb_blk(RK_TRUE);

    // RGA
    rga::settings settings
    {
        ._width = _settings._width,
        ._height = _settings._height,
        ._pixel_format_out = _settings._pixel_format_out,
        ._pixel_format_in = _settings._pixel_format_in,
        ._pixel_format_in_rga = _settings._pixel_format_in_rga,
        ._pixel_format_out_rga = _settings._pixel_format_out_rga,
        ._pixels_per_byte_out = _settings._bytes_per_pixel_out,
        ._pixels_per_byte_in = _settings._bytes_per_pixel_in,
    };
    _rga.set_settings(settings);
    _rga.init(_mb_pool.get_mb_blk(0));

    // VI
    vi::settings vi_settings 
    {
        ._path_to_iq_dir = _settings._path_to_iq_dir,
        ._id_camera = _settings._id_camera,
        ._pixel_format = _settings._pixel_format_in,
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
        ._bytes_per_pixels = _settings._bytes_per_pixel_out,
        ._pixel_format = _settings._pixel_format_out,
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
    int64 last_tick = cv::getTickCount();
    int   frames    = 0;
    double fps      = 0.0;

    while(1)
    {
        _vi.receive_frame_from_channel();
        _rga.process_frame(_vi.get_frame());

    // --- FPS счётчик ---
        frames++;
        int64 now = cv::getTickCount();
        double elapsed = (now - last_tick) / cv::getTickFrequency();
        if (elapsed >= 1.0) {
            fps = frames / elapsed;
            frames = 0;
            last_tick = now;
        }

        void* frame_after_rga = _mb_pool.get_ptr_from_mb_blk(_mb_pool.get_mb_blk(0));
        cv::Mat rgb(_settings._height, _settings._width, CV_8UC3, frame_after_rga);

        char text[32];
        std::snprintf(text, sizeof(text), "FPS: %.1f", fps);
        cv::putText(rgb, text, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    1.0, {0, 0, 0}, 4, cv::LINE_AA);   // обводка
        cv::putText(rgb, text, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    1.0, {0, 255, 0}, 2, cv::LINE_AA); // текст

        RK_MPI_SYS_MmzFlushCache(
            _mb_pool.get_mb_blk(0),
            RK_FALSE
        );

        _venc.exec_frame_from_vi(_vi.get_frame());
        _venc.exec_frame_to_codec();

        _rtsp.send_frame(
            reinterpret_cast<uint8_t*>(_mb_pool.get_ptr_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        );
        _vi.release_frame();
        _venc.release_frame();
    }

    return true;
}

 bool fsm::release()
 {
    _mb_pool.release();
    _rga.release();
    _vi.release();
    _venc.release();
    _rtsp.release();
    RK_MPI_SYS_Exit();

    return true;
 }