#include "br_fsm.h"

#include <cstdio>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

br_fsm::br_fsm()
{

}

br_fsm::~br_fsm()
{

}

bool br_fsm::init()
{
    // MB_POOL
    if (!_mb_pool.init(_settings._mb_blk_count, _settings._width * _settings._height * _settings._bytes_per_pixel_out))
    {
        return false;
    };
    printf("%s: _mb_pool.init: ok\n", __PRETTY_FUNCTION__);

    if (_mb_pool.create_mb_blk(RK_TRUE) == MB_INVALID_HANDLE)
    {
        return false;
    };
    printf("%s: _mb_pool.create_mb_blk: ok\n", __PRETTY_FUNCTION__);

    // MB_POOL под вход модели: размер блока другой, поэтому отдельный пул
    if (!_mb_pool_npu.init(_settings._mb_blk_count, _settings._width_model * _settings._height_model * _settings._bytes_per_pixel_out))
    {
        return false;
    };
    printf("%s: _mb_pool_npu.init: ok\n", __PRETTY_FUNCTION__);

    if (_mb_pool_npu.create_mb_blk(RK_TRUE) == MB_INVALID_HANDLE)
    {
        return false;
    };
    printf("%s: _mb_pool_npu.create_mb_blk: ok\n", __PRETTY_FUNCTION__);

    // RGA: кадр для VENC и кадр под вход модели
    br_rga::settings settings
    {
        ._width = _settings._width,
        ._height = _settings._height,
        ._width_model = _settings._width_model,
        ._height_model = _settings._height_model,
        ._pixel_format_out = _settings._pixel_format_out,
        ._pixel_format_in = _settings._pixel_format_in,
        ._pixel_format_in_rga = _settings._pixel_format_in_rga,
        ._pixel_format_out_rga = _settings._pixel_format_out_rga,
        ._pixels_per_byte_out = _settings._bytes_per_pixel_out,
        ._pixels_per_byte_in = _settings._bytes_per_pixel_in,
    };
    _brga.set_settings(settings);
    if (!_brga.init(_mb_pool.get_mb_blk(0), _mb_pool_npu.get_mb_blk(0)))
    {
        return false;
    };
    printf("%s: _brga.init: ok\n", __PRETTY_FUNCTION__);

    // VI
    br_vi::settings vi_settings 
    {
        ._path_to_iq_dir = _settings._path_to_iq_dir,
        ._id_camera = _settings._id_camera,
        ._pixel_format = _settings._pixel_format_in,
        ._width = _settings._width,
        ._height = _settings._height,
    };
    _vi.set_settings(vi_settings);
    if(!_vi.init())
    {
        return false;
    };
    printf("%s: _vi.init: ok\n", __PRETTY_FUNCTION__);

    // VENC
    br_venc::settings venc_settings
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
    if (!_venc.init(0, _mb_pool.get_mb_blk(0)))
    {
        return false;
    };
    printf("%s: _venc.init: ok\n", __PRETTY_FUNCTION__);

    // RTSP
    br_rtsp::settings rtsp_settings
    {
        ._port = 554,
        ._path = "/live/0",
        ._codec_rtsp = RTSP_CODEC_ID_VIDEO_H264,
    };
    _rtsp.set_settings(rtsp_settings);
    if (!_rtsp.init())
    {
        return false;
    };
    printf("%s: _rtsp.init: ok\n", __PRETTY_FUNCTION__);

    // NPU
    br_npu::settings npu_settings
    {
        ._width = _settings._width,
        ._height = _settings._height,
        ._model_path = _settings._path_to_model,
    };
    _npu.set_settings(npu_settings);
    // Блок под вход модели и RGA-модуль, который в него пишет кадр
    if (!_npu.init(_mb_pool_npu.get_mb_blk(0), &_brga))
    {
        return false;
    };
    printf("%s: _npu.init: ok\n", __PRETTY_FUNCTION__);

    return true;
}

bool br_fsm::start()
{
    bool is_failed;
    while(1)
    {
        is_failed = false;
        // Получаем фрейм с VI
        if (!_vi.receive_frame_from_channel())
        {
            is_failed = true;
        }
        _stage_us.vi += take_stage_time();
        // Обработка в RGA
        if (!is_failed && !_brga.process_frame(_vi.get_last_frame()))
        {
            is_failed = true;
        };
        _stage_us.rga += take_stage_time();
        // Получаем указатель на фрейм после rga
        void* frame_after_brga = _mb_pool.get_ptr_from_mb_blk(_mb_pool.get_mb_blk(0));
        if (!is_failed && frame_after_brga == nullptr)
        {
            is_failed = true;
        }
        // Отдаем кадр в поток детекции: если NPU занят предыдущим кадром, этот пропускается
        if (!is_failed && !_npu.prepare_input(_vi.get_last_frame()))
        {
            is_failed = true;
        }
        _stage_us.npu += take_stage_time();
        // Рисуем рамки по последнему готовому результату детекции
        if (!is_failed && !draw_detections(frame_after_brga))
        {
            is_failed = true;
        }
        _stage_us.draw += take_stage_time();
        // Сбрасываем кэш, иначе мерцает open-cv
        if (!is_failed && !_mb_pool.mmz_flush_cache(0))
        {
            is_failed = true;
        }
        _stage_us.flush += take_stage_time();
        // Отправляем rga frame в venc (кадр уже в mb блоке, с которым работал rga)
        if (!is_failed && !_venc.prepare_frame())
        {
            is_failed = true;
        }
        // Переводим в нужный кодек
        if(!is_failed && !_venc.process_frame())
        {
            is_failed = true;
        }
        _stage_us.venc += take_stage_time();
        // Отправляем в rtsp
        if (!is_failed && !_rtsp.process_frame(
            reinterpret_cast<uint8_t*>(_mb_pool.get_ptr_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        ))
        {
            is_failed = true;
        }
        _stage_us.rtsp += take_stage_time();
        // Очищаем 
        _vi.release_frame();
        _venc.release_codec_frame();
        // Конец кадра: хвост цикла ни в одну стадию не относим
        take_stage_time();
        // Раз в секунду печатаем FPS и средние времена стадий
        print_fps();
    }

    return true;
}

 bool br_fsm::release()
 {
    // Сначала отпускаем всех, кто держит блоки кадра, и только потом сами пулы
    _npu.release();
    _rtsp.release();
    _venc.release();
    _brga.release();
    _vi.release();
    _mb_pool_npu.release();
    _mb_pool.release();
    // Обязательно
    RK_MPI_SYS_Exit();

    return true;
 }

bool br_fsm::draw_detections(void* ptr_frame)
{
    br_npu::object_detect_result_list detections = _npu.get_detections();

    // Кадр после RGA лежит в формате RGB888
    cv::Mat frame(cv::Size(_settings._width, _settings._height), CV_8UC3, ptr_frame);

    for (int i = 0; i < detections.count; i++)
    {
        br_npu::object_detect_result* ptr_result = &(detections.results[i]);

        printf("%s @ (%d %d %d %d) %.3f\n",
               br_npu::get_class_name(ptr_result->cls_id),
               ptr_result->box.left, ptr_result->box.top, ptr_result->box.right, ptr_result->box.bottom,
               ptr_result->prop);

        cv::rectangle(frame,
                      cv::Point(ptr_result->box.left, ptr_result->box.top),
                      cv::Point(ptr_result->box.right, ptr_result->box.bottom),
                      cv::Scalar(0, 255, 0), 3);

        char text[64];
        sprintf(text, "%s %.1f%%", br_npu::get_class_name(ptr_result->cls_id), ptr_result->prop * 100);
        cv::putText(frame, text, cv::Point(ptr_result->box.left, ptr_result->box.top - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    return true;
}

uint64_t br_fsm::get_current_time_us() const
{
    static struct timespec time = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000000 + time.tv_nsec / 1000;
}

uint64_t br_fsm::take_stage_time()
{
    const uint64_t now_us = get_current_time_us();
    const uint64_t delta_us = now_us - _stage_time_us;
    _stage_time_us = now_us;
    return delta_us;
}

void br_fsm::print_fps()
{
    const uint64_t now_us = get_current_time_us();

    if (_fps_start_time_us == 0)
    {
        _fps_start_time_us = now_us;
        return;
    }

    _fps_frames++;

    const uint64_t period_us = now_us - _fps_start_time_us;
    if (period_us < FPS_PERIOD_US)
    {
        return;
    }

    // Средние значения за окно: точнее, чем 1/Δt по одному кадру
    const double frames = (double)_fps_frames;
    const double frame_us = (double)period_us / frames;
    const uint32_t inference_count = _npu.get_inference_count();
    const double det_per_sec = (double)(inference_count - _inference_count_prev) * 1000000.0 / (double)period_us;
    _inference_count_prev = inference_count;

    printf("[fps] %.2f | frame %.1f ms | vi %.1f | rga %.1f | npu %.1f | draw %.1f | flush %.1f | venc %.1f | rtsp %.1f ms | det %.1f/s\n",
           frames * 1000000.0 / (double)period_us,
           frame_us / 1000.0,
           (double)_stage_us.vi / frames / 1000.0,
           (double)_stage_us.rga / frames / 1000.0,
           (double)_stage_us.npu / frames / 1000.0,
           (double)_stage_us.draw / frames / 1000.0,
           (double)_stage_us.flush / frames / 1000.0,
           (double)_stage_us.venc / frames / 1000.0,
           (double)_stage_us.rtsp / frames / 1000.0,
           det_per_sec);

    _stage_us = stage_stats();
    _fps_frames = 0;
    _fps_start_time_us = now_us;
}