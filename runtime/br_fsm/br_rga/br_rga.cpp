#include "br_rga.h"
#include <cstddef>

// Цвет полей кадра входа модели (114, 114, 114) — такой же, как при обучении модели
// TODO: переделать на настройки.
constexpr int PADDING_COLOR = 0x727272;

br_rga::br_rga()
{

}

br_rga::~br_rga()
{

}

bool br_rga::init(MB_BLK ptr_blk, MB_BLK ptr_blk_model)
{
    // Считаем размеры уходящего буфера в venc
    _size_for_venc = _settings._width * _settings._height * _settings._pixels_per_byte_out;
    // Считаем размер буфера входа модели
    _size_for_model = _settings._width_model * _settings._height_model * _settings._pixels_per_byte_out;
    // Считаем размеры приходящего буфера с vi
    _size_from_vi = _settings._width * _settings._height * _settings._pixels_per_byte_in;
    // Считаем размер кадра под внешний скрипт: разрешение кадра, три байта на пиксель
    _size_for_hook = _settings._width * _settings._height * _settings._pixels_per_byte_out;

    // Вписываем кадр во вход модели с сохранением пропорций (letterbox): масштаб по меньшей
    // стороне, остаток — поля вокруг кадра
    const float scale_x = (float)_settings._width_model / (float)_settings._width;
    const float scale_y = (float)_settings._height_model / (float)_settings._height;
    _scale = scale_x < scale_y ? scale_x : scale_y;
    _content_w = (uint16_t)((float)_settings._width * _scale + 0.5f);
    _content_h = (uint16_t)((float)_settings._height * _scale + 0.5f);
    _pad_x = (_settings._width_model - _content_w) / 2;
    _pad_y = (_settings._height_model - _content_h) / 2;
    printf("%s: letterbox %ux%u -> %ux%u в %ux%u, поля (%u, %u)\n",
           __PRETTY_FUNCTION__, _settings._width, _settings._height, _content_w, _content_h,
           _settings._width_model, _settings._height_model, _pad_x, _pad_y);

    // Прокидываем блок памяти фрейма в RGA и получаем указатель на буфер
    _handle_buffer_venc = importbuffer_fd(RK_MPI_MB_Handle2Fd(ptr_blk), _size_for_venc);
    if (_handle_buffer_venc == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    memset(&_buffer_for_venc, 0, sizeof(_buffer_for_venc));
    // Упаковываем буфер из RGA и получаем сам буфер
    _buffer_for_venc = wrapbuffer_handle(_handle_buffer_venc, _settings._width, _settings._height, _settings._pixel_format_out_rga);
    _buffer_for_venc.wstride = _settings._width;
    _buffer_for_venc.height = _settings._height;

    // То же для буфера входа модели
    _handle_buffer_model = importbuffer_fd(RK_MPI_MB_Handle2Fd(ptr_blk_model), _size_for_model);
    if (_handle_buffer_model == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    memset(&_buffer_for_model, 0, sizeof(_buffer_for_model));
    _buffer_for_model = wrapbuffer_handle(_handle_buffer_model, _settings._width_model, _settings._height_model, _settings._pixel_format_out_rga);
    _buffer_for_model.wstride = _settings._width_model;
    _buffer_for_model.height = _settings._height_model;

    return true;
}

void br_rga::set_settings(const settings settings)
{
    _settings = settings;
}

float br_rga::get_scale() const
{
    return _scale;
}

uint16_t br_rga::get_pad_x() const
{
    return _pad_x;
}

uint16_t br_rga::get_pad_y() const
{
    return _pad_y;
}

bool br_rga::process_frame_model(VIDEO_FRAME_INFO_S* ptr_frame)
{
    int ret = 0;

    // Получаем handle на блок памяти фрейма
    int fd_from_vi = RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk);

    if (fd_from_vi < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Прокидываем блок памяти фрейма в RGA и получаем указатель на буфер
    rga_buffer_handle_t handle_buffer_vi = importbuffer_fd(fd_from_vi, _size_from_vi);
    if (handle_buffer_vi == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Упаковываем буфер из RGA и получаем сам буфер
    rga_buffer_t buffer_from_vi = wrapbuffer_handle(
        handle_buffer_vi,
        _settings._width,
        _settings._height,
        _settings._pixel_format_in_rga
    );

    // Заливаем вход модели цветом полей, чтобы поля не собирали остатки прошлых кадров и рамок
    im_rect rect_model { 0, 0, (int)_settings._width_model, (int)_settings._height_model };
    ret = imfill(_buffer_for_model, rect_model, PADDING_COLOR);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: imfill fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        releasebuffer_handle(handle_buffer_vi);
        return false;
    }

    // Масштабируем кадр, конвертируем цвет и кладем его по центру: все одним проходом RGA
    im_rect rect_from_vi { 0, 0, (int)_settings._width, (int)_settings._height };
    im_rect rect_content { (int)_pad_x, (int)_pad_y, (int)_content_w, (int)_content_h };
    ret = improcess(buffer_from_vi, _buffer_for_model, {}, rect_from_vi, rect_content, {}, -1, nullptr, nullptr, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: improcess fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        releasebuffer_handle(handle_buffer_vi);
        return false;
    }

    ret = releasebuffer_handle(handle_buffer_vi);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s, ret: %d\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret), ret);
        return false;
    }

    return true;
}

bool br_rga::process_frame(VIDEO_FRAME_INFO_S* ptr_frame)
{
    int ret = 0;

    // Получаем handle на блок памяти фрейма
	int fd_from_vi = RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk);

    if (fd_from_vi < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Прокидываем блок памяти фрейма в RGA и получаем указатель на буфер
    rga_buffer_handle_t handle_buffer_vi = importbuffer_fd(fd_from_vi, _size_from_vi);
    if (handle_buffer_vi == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Упаковываем буфер из RGA и получаем сам буфер
    rga_buffer_t buffer_from_vi = wrapbuffer_handle(
        handle_buffer_vi, 
        _settings._width, 
        _settings._height,
        _settings._pixel_format_in_rga
    );

    // Проверяем буферы vi и venc
    ret = imcheck(buffer_from_vi, _buffer_for_venc, {}, {});
    if (ret != IM_STATUS_NOERROR) 
    {
        printf("%s: imcheck fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        ret = releasebuffer_handle(handle_buffer_vi);
        if (ret != IM_STATUS_NOERROR)
        {
            printf("%s: imcheck&releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        }
        return false;
    } 

    // Конвертируем
    ret = imcvtcolor(
        buffer_from_vi, 
        _buffer_for_venc, 
        _settings._pixel_format_in_rga,
        _settings._pixel_format_out_rga, 
        1
    );
    if (ret != IM_STATUS_SUCCESS) 
    {
        printf("%s: imcvtcolor fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        ret = releasebuffer_handle(handle_buffer_vi);
        if (ret != IM_STATUS_NOERROR)
        {
            printf("%s: imcheck&releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        }
        return false;
    }

    ret = releasebuffer_handle(handle_buffer_vi);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s, ret: %d\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret), ret);
        return false;
    }

    return true;
}

bool br_rga::process_frame_hook(VIDEO_FRAME_INFO_S* ptr_frame, MB_BLK ptr_blk_hook)
{
    int ret = 0;

    // Кадр с VI и буфер скрипта импортируем на каждый кадр: так же, как в остальных проходах,
    // хендлы живут только до конца конвертации
    int fd_from_vi = RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk);
    if (fd_from_vi < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    int fd_for_hook = RK_MPI_MB_Handle2Fd(ptr_blk_hook);
    if (fd_for_hook < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    rga_buffer_handle_t handle_buffer_vi = importbuffer_fd(fd_from_vi, _size_from_vi);
    rga_buffer_handle_t handle_buffer_hook = importbuffer_fd(fd_for_hook, _size_for_hook);
    if (handle_buffer_vi == 0 || handle_buffer_hook == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        if (handle_buffer_vi != 0)
        {
            releasebuffer_handle(handle_buffer_vi);
        }
        if (handle_buffer_hook != 0)
        {
            releasebuffer_handle(handle_buffer_hook);
        }
        return false;
    }

    rga_buffer_t buffer_from_vi = wrapbuffer_handle(
        handle_buffer_vi,
        _settings._width,
        _settings._height,
        _settings._pixel_format_in_rga
    );
    rga_buffer_t buffer_for_hook = wrapbuffer_handle(
        handle_buffer_hook,
        _settings._width,
        _settings._height,
        _settings._pixel_format_hook_rga
    );

    // Кадр для скрипта: то же разрешение и та же конвертация из YUV, что и у кадра в поток,
    // поэтому один проход RGA без масштабирования
    ret = imcvtcolor(
        buffer_from_vi,
        buffer_for_hook,
        _settings._pixel_format_in_rga,
        _settings._pixel_format_hook_rga,
        1
    );
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: imcvtcolor fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        releasebuffer_handle(handle_buffer_vi);
        releasebuffer_handle(handle_buffer_hook);
        return false;
    }

    ret = releasebuffer_handle(handle_buffer_vi);
    ret = releasebuffer_handle(handle_buffer_hook);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s, ret: %d\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret), ret);
        return false;
    }

    return true;
}

bool br_rga::process_frame_from_hook(MB_BLK ptr_blk_hook)
{
    int ret = 0;

    // Размеченный кадр ложится в буфер VENC копией: буфер скрипта нужен ему же под следующий
    // кадр, поэтому отдать блок скрипта в VENC напрямую нельзя
    int fd_for_hook = RK_MPI_MB_Handle2Fd(ptr_blk_hook);
    if (fd_for_hook < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    rga_buffer_handle_t handle_buffer_hook = importbuffer_fd(fd_for_hook, _size_for_hook);
    if (handle_buffer_hook == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    rga_buffer_t buffer_for_hook = wrapbuffer_handle(
        handle_buffer_hook,
        _settings._width,
        _settings._height,
        _settings._pixel_format_hook_rga
    );

    // Форматы совпадают, поэтому это копия, а не конвертация: цветовые проходы RGA тут не нужны
    ret = imcopy(buffer_for_hook, _buffer_for_venc);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: imcopy fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        releasebuffer_handle(handle_buffer_hook);
        return false;
    }

    ret = releasebuffer_handle(handle_buffer_hook);
    if (ret != IM_STATUS_SUCCESS)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s, ret: %d\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret), ret);
        return false;
    }

    return true;
}

bool br_rga::release()
{
    int ret = 0;

    if (_handle_buffer_model != 0)
    {
        ret = releasebuffer_handle(_handle_buffer_model);
        if (ret != IM_STATUS_NOERROR)
        {
            printf("%s: releasebuffer_handle model fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
            return false;
        }
        _handle_buffer_model = 0;
    }

    ret = releasebuffer_handle(_handle_buffer_venc);
    if (ret != IM_STATUS_NOERROR)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        return false;
    }
    _handle_buffer_venc = 0;

    return true;
}
