#include "br_rga.h"
#include <cstddef>

br_rga::br_rga()
{

}

br_rga::~br_rga()
{

}

bool br_rga::init(MB_BLK ptr_blk)
{
    // Считаем размеры уходящего буфера в venc
    _size_for_venc = _settings._width_out * _settings._height_out * _settings._pixels_per_byte_out;
    // Считаем размеры приходящего буфера с vi
    _size_from_vi = _settings._width * _settings._height * _settings._pixels_per_byte_in;

    // Кадр обрезаем по центру под пропорции выхода, чтобы выход заполнился целиком и без искажений
    const float scale_x = (float)_settings._width_out / (float)_settings._width;
    const float scale_y = (float)_settings._height_out / (float)_settings._height;
    const float scale = scale_x > scale_y ? scale_x : scale_y;
    _crop_w = (uint16_t)((float)_settings._width_out / scale + 0.5f);
    _crop_h = (uint16_t)((float)_settings._height_out / scale + 0.5f);
    _crop_x = (_settings._width - _crop_w) / 2;
    _crop_y = (_settings._height - _crop_h) / 2;
    printf("%s: crop (%u, %u) %ux%u -> %ux%u\n",
           __PRETTY_FUNCTION__, _crop_x, _crop_y, _crop_w, _crop_h, _settings._width_out, _settings._height_out);

    // Прокидываем блок памяти фрейма в RGA и получаем указатель на буфер
    _handle_buffer_venc = importbuffer_fd(RK_MPI_MB_Handle2Fd(ptr_blk), _size_for_venc);
    if (_handle_buffer_venc == 0)
    {
        printf("%s: importbuffer_fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    memset(&_buffer_for_venc, 0, sizeof(_buffer_for_venc));
    // Упаковываем буфер из RGA и получаем сам буфер
    _buffer_for_venc = wrapbuffer_handle(_handle_buffer_venc, _settings._width_out, _settings._height_out, _settings._pixel_format_out_rga);
    _buffer_for_venc.wstride = _settings._width_out;
    _buffer_for_venc.height = _settings._height_out;

    return true;
}

void br_rga::set_settings(const settings settings)
{
    _settings = settings;
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

    // Обрезаем кадр по центру, масштабируем под выход и конвертируем цвет одним проходом RGA
    im_rect rect_from_vi { _crop_x, _crop_y, _crop_w, _crop_h };
    im_rect rect_to_venc { 0, 0, _settings._width_out, _settings._height_out };
    ret = improcess(buffer_from_vi, _buffer_for_venc, {}, rect_from_vi, rect_to_venc, {}, -1, nullptr, nullptr, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) 
    {
        printf("%s: improcess fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
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

bool br_rga::release()
{
    int ret = releasebuffer_handle(_handle_buffer_venc);
    if (ret != IM_STATUS_NOERROR)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        return false;
    }
    return true;
}
