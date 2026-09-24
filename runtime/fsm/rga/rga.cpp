#include "rga.h"
#include <cstddef>

rga::rga()
{

}

rga::~rga()
{

}

bool rga::init()
{
    _wstride  = (_settings._width + 15) & ~15;   
    _bgr_size = _wstride * _settings._height * 3;

    return false;
}

void rga::set_settings(const settings settings)
{
    _settings = settings;
}

bool rga::create_buffer(MB_BLK ptr_blk)
{
    dst_fd = RK_MPI_MB_Handle2Fd(ptr_blk);

    return true;
}

int rga::process_frame(VIDEO_FRAME_INFO_S* ptr_frame)
{
    void *vi_data = RK_MPI_MB_Handle2VirAddr(ptr_frame->stVFrame.pMbBlk);
	int vi_fd = RK_MPI_MB_Handle2Fd(ptr_frame->stVFrame.pMbBlk);

    int src_size = _wstride * _settings._height * 3 / 2;
    int dst_size = _wstride * _settings._height * 3;

    rga_buffer_handle_t src_handle = importbuffer_fd(vi_fd, src_size);
    rga_buffer_handle_t dst_handle = importbuffer_fd(dst_fd, dst_size);

    if (src_handle && dst_handle) {
        rga_buffer_t src = wrapbuffer_handle(src_handle, _settings._width, _settings._height,
                                            RK_FORMAT_YCbCr_420_SP);
        rga_buffer_t dst = wrapbuffer_handle(dst_handle, _settings._width, _settings._height,
                                            RK_FORMAT_BGR_888);

        src.wstride = _wstride;
        src.hstride = _settings._height;
        dst.wstride = _wstride;
        dst.hstride = _settings._height;

        int check_ret = imcheck(src, dst, {}, {});
        if (IM_STATUS_NOERROR != check_ret) {
            printf("RGA imcheck error: %s\n", imStrError((IM_STATUS)check_ret));
        } else {
            int ret = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP,
                                RK_FORMAT_BGR_888, 1);
            if (ret != IM_STATUS_SUCCESS) {
                printf("RGA imcvtcolor failed: %s\n", imStrError((IM_STATUS)ret));
            }
        }

        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
    }
    return 0;
}

uint32_t rga::get_wstride() const
{
    return _wstride;
}

uint32_t rga::get_bgr_size() const
{
    return _bgr_size;
}

