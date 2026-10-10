#include "br_fsm.h"

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Пересчет координаты из letterbox обратно в кадр
static int map_coordinate(int value, int padding, float scale)
{
    return (int)((float)(value - padding) / scale);
}

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

    // RGA
    br_rga::settings settings
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
    _brga.set_settings(settings);
    if (!_brga.init(_mb_pool.get_mb_blk(0)))
    {
        return false;
    };
    printf("%s: _brga.init: ok\n", __PRETTY_FUNCTION__);

    // NPU
    br_npu::settings npu_settings
    {
        ._model_path = _settings._path_to_model,
        ._label_path = _settings._path_to_labels,
    };
    _npu.set_settings(npu_settings);
    if (!_npu.init())
    {
        return false;
    };
    printf("%s: _npu.init: ok\n", __PRETTY_FUNCTION__);

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
        // Обработка в RGA
        if (!is_failed && !_brga.process_frame(_vi.get_last_frame()))
        {
            is_failed = true;
        };
        // Получаем указатель на фрейм после rga
        void* frame_after_brga = _mb_pool.get_ptr_from_mb_blk(_mb_pool.get_mb_blk(0));
        if (!is_failed && frame_after_brga == nullptr)
        {
            is_failed = true;
        }
        // Сбрасываем кэш, иначе мерцает open-cv
        if (!is_failed && !_mb_pool.mmz_flush_cache(0))
        {
            is_failed = true;
        }
        // NPU: детекция и отрисовка рамок через opencv прямо в кадре после RGA
        if (!is_failed && !process_frame_with_opencv(reinterpret_cast<uint8_t*>(frame_after_brga)))
        {
            is_failed = true;
        }
        // После записи из cpu (отрисовка рамок) сбрасываем кэш, чтобы venc по dma увидел изменения
        if (!is_failed && !_mb_pool.mmz_flush_cache(0))
        {
            is_failed = true;
        }
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
        // Отправляем в rtsp
        if (!is_failed && !_rtsp.process_frame(
            reinterpret_cast<uint8_t*>(_mb_pool.get_ptr_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        ))
        {
            is_failed = true;
        }
        // Очищаем 
        _vi.release_frame();
        _venc.release_codec_frame();
    }

    return true;
}

bool br_fsm::process_frame_with_opencv(uint8_t* ptr_data)
{
    if (ptr_data == nullptr)
    {
        printf("%s: ptr_data is null!\n", __PRETTY_FUNCTION__);
        return false;
    }

    int model_width = _npu.get_model_width();
    int model_height = _npu.get_model_height();

    // Кадр после RGA уже в RGB888, оборачиваем его без копирования и рисуем рамки прямо в него
    cv::Mat frame(_settings._height, _settings._width, CV_8UC3, ptr_data);

    // Letterbox: вписываем кадр в квадрат модели, сохраняя пропорции
    float scale_x = (float)model_width / (float)_settings._width;
    float scale_y = (float)model_height / (float)_settings._height;
    float scale = scale_x < scale_y ? scale_x : scale_y;

    int input_width = (int)((float)_settings._width * scale);
    int input_height = (int)((float)_settings._height * scale);
    int left_padding = (model_width - input_width) / 2;
    int top_padding = (model_height - input_height) / 2;

    cv::Mat input_scale;
    cv::resize(frame, input_scale, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat letterbox_image(model_height, model_width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi(left_padding, top_padding, input_width, input_height);
    input_scale.copyTo(letterbox_image(roi));

    // RGA отдает RGB888, а модель ждет BGR (в примере кадр уходит в модель прямо из cv::cvtColor(...2BGR))
    cv::Mat model_input;
    cv::cvtColor(letterbox_image, model_input, cv::COLOR_RGB2BGR);

    // NPU
    object_detect_result_list od_results;
    if (!_npu.inference(model_input.data, &od_results))
    {
        return false;
    }

    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result* ptr_det_result = &(od_results.results[i]);

        // Возвращаем координаты из letterbox обратно в кадр
        int s_x = map_coordinate((int)(ptr_det_result->box.left), left_padding, scale);
        int s_y = map_coordinate((int)(ptr_det_result->box.top), top_padding, scale);
        int e_x = map_coordinate((int)(ptr_det_result->box.right), left_padding, scale);
        int e_y = map_coordinate((int)(ptr_det_result->box.bottom), top_padding, scale);

        printf("%s @ (%d %d %d %d) %.3f\n", _npu.class_name(ptr_det_result->cls_id), s_x, s_y, e_x, e_y, ptr_det_result->prop);

        cv::rectangle(frame, cv::Point(s_x, s_y), cv::Point(e_x, e_y), cv::Scalar(0, 255, 0), 3);

        char text[OBJ_NAME_MAX_SIZE + 16];
        snprintf(text, sizeof(text), "%s %.1f%%", _npu.class_name(ptr_det_result->cls_id), ptr_det_result->prop * 100);
        cv::putText(frame, text, cv::Point(s_x, s_y - 8), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    return true;
}

 bool br_fsm::release()
 {
    _mb_pool.release();
    _brga.release();
    _npu.release();
    _vi.release();
    _venc.release();
    _rtsp.release();
    // Обязательно
    RK_MPI_SYS_Exit();

    return true;
 }