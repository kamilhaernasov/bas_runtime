#include "rga.h"
#include <cstddef>

rga::rga()
{

}

rga::~rga()
{

}

bool rga::init(MB_BLK ptr_blk)
{
    // Считаем размеры уходящего буфера в venc
    _size_for_venc = _settings._width * _settings._height * _settings._pixels_per_byte_out;
    // Считаем размеры приходящего буфера с vi
    _size_from_vi = _settings._width * _settings._height * _settings._pixels_per_byte_in;
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

    return true;
}

void rga::set_settings(const settings settings)
{
    _settings = settings;
}

bool rga::process_frame(VIDEO_FRAME_INFO_S* ptr_frame)
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

bool rga::release()
{
    int ret = releasebuffer_handle(_handle_buffer_venc);
    if (ret != IM_STATUS_NOERROR)
    {
        printf("%s: releasebuffer_handle fail! Desc: %s\n", __PRETTY_FUNCTION__, imStrError((IM_STATUS)ret));
        return false;
    }
    return true;
}
