#include "br_fsm.h"

#include <cstdio>
#include <iostream>

namespace
{
    /* Кадр под внешний скрипт - три байта на пиксель, формат тот же, что уходит в поток */
    constexpr uint32_t HOOK_BYTES_PER_PIXEL = 3;
}

br_fsm::br_fsm()
{
    std::cout << __PRETTY_FUNCTION__;
}

br_fsm::~br_fsm()
{
    std::cout << __PRETTY_FUNCTION__;
}

void br_fsm::set_settings(const settings settings)
{
    std::cout << __PRETTY_FUNCTION__;
    _settings = settings;
}

bool br_fsm::init()
{
    std::cout << __PRETTY_FUNCTION__;
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

    // MB_POOL под кадр внешнего скрипта: формат тот же, что уходит в поток, то есть три байта на пиксель
    if (!_mb_pool_hook.init(_settings._mb_blk_count, _settings._width * _settings._height * HOOK_BYTES_PER_PIXEL))
    {
        return false;
    };
    printf("%s: _mb_pool_hook.init: ok\n", __PRETTY_FUNCTION__);

    if (_mb_pool_hook.create_mb_blk(RK_TRUE) == MB_INVALID_HANDLE)
    {
        return false;
    };
    printf("%s: _mb_pool_hook.create_mb_blk: ok\n", __PRETTY_FUNCTION__);

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
        ._pixel_format_hook_rga = _settings._pixel_format_hook_rga,
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
        ._bitrate = _settings._venc_bitrate == 0 ? _settings._width * _settings._height / 8 * 30 : _settings._venc_bitrate,
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
        ._port = _settings._rtsp_port,
        ._path = _settings._rtsp_path,
        ._codec_rtsp = RTSP_CODEC_ID_VIDEO_H264,
    };
    _rtsp.set_settings(rtsp_settings);
    if (!_rtsp.init())
    {
        return false;
    };
    printf("%s: _rtsp.init: ok\n", __PRETTY_FUNCTION__);

    // Model
    _model = br_model_create(_settings._model_type);
    if (_model == nullptr)
    {
        return false;
    }

    br_model::settings model_settings
    {
        ._path_to_model = _settings._path_to_model,
        ._width_model = _settings._width_model,
        ._height_model = _settings._height_model,
        ._box_thresh = _settings._box_thresh,
        ._nms_thresh = _settings._nms_thresh,
        ._max_detections = _settings._max_detections,
        ._path_to_labels = _settings._path_to_labels,
        ._path_to_anchors = _settings._path_to_anchors,
        ._width = _settings._width,
        ._height = _settings._height,
        ._scale = _brga.get_scale(),
        ._pad_x = _brga.get_pad_x(),
        ._pad_y = _brga.get_pad_y(),
    };
    _model->set_settings(model_settings);

    // Блок под вход модели и разбор ее выходов: в блок пишет RGA, из него читает NPU.
    // Свои настройки (файл модели и размер входа) NPU берет у самой модели
    if (!_npu.init(_mb_pool_npu.get_mb_blk(0), _model.get()))
    {
        return false;
    };
    printf("%s: _npu.init: ok\n", __PRETTY_FUNCTION__);

    // Хук: внешний скрипт пользователя. Кадр под него готовит RGA, рисует он сам, своей
    // отрисовки в рантайме нет. Настройки лежат отдельно (hook.ini)
    br_hook::settings hook_settings
    {
        ._is_enabled = _settings._hook_enabled,
        ._is_async = _settings._hook_async,
        ._path_script = _settings._path_to_hook,
        ._path_python = _settings._path_python,
        ._timeout_ms = _settings._hook_timeout_ms,
        ._stuck_ms = _settings._hook_stuck_ms,
        ._start_ms = _settings._hook_start_ms,
    };
    _hook.set_settings(hook_settings);
    if (!_hook.init(_settings._width, _settings._height, _mb_pool_hook.get_mb_blk(0)))
    {
        return false;
    };
    printf("%s: _hook.init: ok\n", __PRETTY_FUNCTION__);

    return true;
}

bool br_fsm::start()
{
    bool is_failed;
    while(1)
    {
        is_failed = false;
        // Времена стадий этого кадра: та же дельта уходит и в накопление за окно (print_fps)
        stage_stats frame_us;
        // Получаем фрейм с VI
        if (!_vi.receive_frame_from_channel())
        {
            is_failed = true;
        }
        _stage_us.vi += (frame_us.vi = take_stage_time());
        // Хук, часть первая: забираем готовый кадр скрипта до конвертации кадра с VI - место
        // в буфере VENC все равно займет он, и проход RGA можно не тратить
        uint64_t hook_us = take_stage_time();
        bool is_hook_frame = false;
        if (!is_failed && _hook.is_enabled())
        {
            _hook.poll();
            if (_hook.take_ready())
            {
                // Кадр писал скрипт процессором, а читать его дальше будет железо: без сброса
                // кэша разметка может до железа не дойти
                if (_mb_pool_hook.mmz_flush_cache(0))
                {
                    // Разметка ложится в буфер VENC копией. Отдать блок скрипта в VENC напрямую
                    // нельзя: этим же блоком скрипт получит следующий кадр, пока VENC еще читает
                    // текущий. Копия тут - развязка двух буферов
                    is_hook_frame = _brga.process_frame_from_hook(_mb_pool_hook.get_mb_blk(0));
                }
            }
        }
        hook_us += take_stage_time();
        // Обработка в RGA
        if (!is_failed && !is_hook_frame && !_brga.process_frame(_vi.get_last_frame()))
        {
            is_failed = true;
        };
        _stage_us.rga += (frame_us.rga = take_stage_time());
        // Отдаем кадр в поток детекции. Спрашиваем NPU до прохода RGA: если он занят
        // предыдущим кадром, кадр пропускается и вписывание в вход модели не тратится
        if (!is_failed && _npu.is_input_free())
        {
            // Кадр под вход модели укладывает RGA, и только потом вход уходит на прогон
            if (!_brga.process_frame_model(_vi.get_last_frame()))
            {
                is_failed = true;
            }
            else
            {
                _npu.submit_input();
            }
        }
        _stage_us.npu += (frame_us.npu = take_stage_time());
        // Хук, часть вторая: отдаем скрипту текущий кадр. Пока он с ним работает, кадр ему
        // не отдаем: в поток идут текущие кадры с камеры
        if (!is_failed && _hook.is_enabled() && _hook.is_free())
        {
            if (_brga.process_frame_hook(_vi.get_last_frame(), _mb_pool_hook.get_mb_blk(0)))
            {
                // Кадр в буфер писала железка (RGA), а читает его скрипт процессором: без сброса
                // кэша скрипт прочитает то, что осталось в кэше с прошлого кадра
                _mb_pool_hook.mmz_invalidate_cache(0);
                // Боксы уезжают с каждым кадром - последние полученные от NPU. Отдавать их только
                // к свежему прогону нельзя: прогон не успевает за кадрами, и разметка мерцала бы
                _hook.submit(_npu.get_detections());
                // Синхронный режим: конвеер ждет скрипт, но не дольше своего бюджета.
                // Не дождались — в поток уйдет кадр с камеры, он уже готов
                if (!_hook.is_async() && _hook.wait())
                {
                    if (_mb_pool_hook.mmz_flush_cache(0))
                    {
                        // Разметка того же кадра ложится в буфер VENC поверх кадра с VI
                        _brga.process_frame_from_hook(_mb_pool_hook.get_mb_blk(0));
                    }
                }
            }
        }
        _stage_us.hook += (frame_us.hook = hook_us + take_stage_time());
        // Сбрасываем кэш после RGA: иначе VENC прочитает не то, что лежит в памяти
        if (!is_failed && !_mb_pool.mmz_flush_cache(0))
        {
            is_failed = true;
        }
        _stage_us.flush += (frame_us.flush = take_stage_time());
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
        _stage_us.venc += (frame_us.venc = take_stage_time());
        // Отправляем в rtsp
        if (!is_failed && !_rtsp.process_frame(
            reinterpret_cast<uint8_t*>(_mb_pool.get_ptr_from_mb_blk(_venc.get_codec_frame()->pstPack->pMbBlk)), 
            _venc.get_codec_frame()->pstPack->u32Len,
            _venc.get_codec_frame()->pstPack->u64PTS
        ))
        {
            is_failed = true;
        }
        _stage_us.rtsp += (frame_us.rtsp = take_stage_time());
        // Очищаем 
        _vi.release_frame();
        _venc.release_codec_frame();
        // Конец кадра: хвост цикла ни в одну стадию не относим
        take_stage_time();
        // Раз в секунду печатаем FPS и средние времена стадий
        print_fps();
        // Печатается только под debug_frame: времена стадий этого кадра и число детекций
        print_frame_debug(frame_us);
    }

    return true;
}

 bool br_fsm::release()
 {
    // Сначала внешний скрипт: он читает свой буфер, который мы дальше и освободим
    _hook.release();
    // Отпускаем всех, кто держит блоки кадра, и только потом сами пулы
    _npu.release();
    // Разбор модели больше не нужен: NPU уже остановлен
    _model.reset();
    _rtsp.release();
    _venc.release();
    _brga.release();
    _vi.release();
    _mb_pool_npu.release();
    _mb_pool.release();
    _mb_pool_hook.release();
    // Обязательно
    RK_MPI_SYS_Exit();

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

    printf("[fps] %.2f | frame %.1f ms | vi %.1f | rga %.1f | npu %.1f | hook %.1f | flush %.1f | venc %.1f | rtsp %.1f ms | det %.1f/s\n",
           frames * 1000000.0 / (double)period_us,
           frame_us / 1000.0,
           (double)_stage_us.vi / frames / 1000.0,
           (double)_stage_us.rga / frames / 1000.0,
           (double)_stage_us.npu / frames / 1000.0,
           (double)_stage_us.hook / frames / 1000.0,
           (double)_stage_us.flush / frames / 1000.0,
           (double)_stage_us.venc / frames / 1000.0,
           (double)_stage_us.rtsp / frames / 1000.0,
           det_per_sec);

    _stage_us = stage_stats();
    _fps_frames = 0;
    _fps_start_time_us = now_us;
}

void br_fsm::print_frame_debug(const stage_stats& frame_us)
{
#ifdef debug_frame
    // Сколько рамок пришло с последнего готового прогона: их же рисует скрипт хука
    const uint32_t detection_count = (uint32_t)_npu.get_detections().size();

    printf("[frame] vi %.2f | rga %.2f | npu %.2f | hook %.2f | flush %.2f | venc %.2f | rtsp %.2f ms | det %u\n",
           (double)frame_us.vi / 1000.0,
           (double)frame_us.rga / 1000.0,
           (double)frame_us.npu / 1000.0,
           (double)frame_us.hook / 1000.0,
           (double)frame_us.flush / 1000.0,
           (double)frame_us.venc / 1000.0,
           (double)frame_us.rtsp / 1000.0,
           detection_count);
#endif
}