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
    // Ширина с выравниванием под 16 байт
    _buffer_size = _settings._width * _settings._height * _settings._pixels_per_byte_out;

    _size_for_venc = _settings._width * _settings._height * _settings._pixels_per_byte_out;
    _size_from_vi = _settings._width * _settings._height * _settings._pixels_per_byte_in;

    _handle_buffer_venc = importbuffer_fd(RK_MPI_MB_Handle2Fd(ptr_blk), _size_for_venc);

    _buffer_for_venc = wrapbuffer_handle(_handle_buffer_venc, _settings._width, _settings._height, _settings._pixel_format_out_rga);
    _buffer_for_venc.wstride = _settings._width;
    _buffer_for_venc.height = _settings._height;

    return false;
}

void rga::set_settings(const settings settings)
{
    _settings = settings;
}

int rga::process_frame(VIDEO_FRAME_INFO_S* ptr_frame)
{
    void *vi_ptr_data = RK_MPI_MB_Handle2VirAddr(ptr_frame->stVFrame.pMbBlk);
	int fd_from_vi = RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk);

    rga_buffer_handle_t handle_buffer_vi = importbuffer_fd(fd_from_vi, _size_from_vi);

    if (handle_buffer_vi) {
        rga_buffer_t buffer_from_vi = wrapbuffer_handle(
            handle_buffer_vi, 
            _settings._width, 
            _settings._height,
            _settings._pixel_format_in_rga
        );

        buffer_from_vi.wstride = _settings._width;
        buffer_from_vi.hstride = _settings._height;

        int check_ret = imcheck(buffer_from_vi, _buffer_for_venc, {}, {});
        if (IM_STATUS_NOERROR != check_ret) 
        {
            printf("RGA imcheck error: %s\n", imStrError((IM_STATUS)check_ret));
        } 
        else 
        {
            int ret = imcvtcolor(
                buffer_from_vi, 
                _buffer_for_venc, 
                _settings._pixel_format_in_rga,
                _settings._pixel_format_out_rga, 
                1
            );

            if (ret != IM_STATUS_SUCCESS) 
            {
                printf("RGA imcvtcolor failed: %s\n", imStrError((IM_STATUS)ret));
            }
        }

        releasebuffer_handle(handle_buffer_vi);
    }
    return 0;
}

void rga::release()
{
    releasebuffer_handle(_handle_buffer_venc);
}
