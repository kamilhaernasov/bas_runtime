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

class br_rga
{
public:
    struct settings
    {
        uint16_t _width;
        uint16_t _height;
        uint16_t _width_model;
        uint16_t _height_model;
        PIXEL_FORMAT_E _pixel_format_out;
        PIXEL_FORMAT_E _pixel_format_in;
        _Rga_SURF_FORMAT _pixel_format_in_rga;
        _Rga_SURF_FORMAT _pixel_format_out_rga;
        // Формат кадра-копии для внешнего скрипта: RGB888 - тот же, что уходит в VENC
        _Rga_SURF_FORMAT _pixel_format_hook_rga;
        float _pixels_per_byte_out;
        float _pixels_per_byte_in;
    };

    explicit br_rga();
    ~br_rga();

    bool init(MB_BLK ptr_blk, MB_BLK ptr_blk_model);
    void set_settings(const settings settings);

    // Кадр для VENC: конвертация один в один
    bool process_frame(VIDEO_FRAME_INFO_S* ptr_frame);
    // Кадр для входа модели: вписывается с сохранением пропорций, поля заливаются
    bool process_frame_model(VIDEO_FRAME_INFO_S* ptr_frame);
    // Кадр для внешнего скрипта: то же разрешение и тот же формат, что и у кадра в поток
    bool process_frame_hook(VIDEO_FRAME_INFO_S* ptr_frame, MB_BLK ptr_blk_hook);
    // Возврат кадра из скрипта в поток: копия в буфер VENC, форматы совпадают
    bool process_frame_from_hook(MB_BLK ptr_blk_hook);

    bool release();

    // Геометрия вписывания кадра во вход модели
    float get_scale() const;
    uint16_t get_pad_x() const;
    uint16_t get_pad_y() const;
private:
    settings _settings;

    uint32_t _size_for_venc;
    uint32_t _size_for_model;
    uint32_t _size_from_vi;
    // Кадр под внешний скрипт: то же разрешение и те же три байта на пиксель
    uint32_t _size_for_hook;

    // Прямоугольник входа модели, в который попадает кадр, и поля вокруг него
    float _scale = 0;
    uint16_t _content_w = 0;
    uint16_t _content_h = 0;
    uint16_t _pad_x = 0;
    uint16_t _pad_y = 0;

    rga_buffer_handle_t _handle_buffer_venc = 0;
    rga_buffer_t _buffer_for_venc {};
    rga_buffer_handle_t _handle_buffer_model = 0;
    rga_buffer_t _buffer_for_model {};
};