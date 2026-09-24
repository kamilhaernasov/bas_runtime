#pragma once

#include <cstdint>
#include <vector>

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

class mb_pool
{
public:
    explicit mb_pool();
    ~mb_pool();

    void init(const uint8_t countBlk, const uint16_t width, const uint16_t height, const uint16_t bytesPerPixel);
    void release();

    MB_BLK create_mb_blk(const RK_BOOL block);
    MB_BLK get_mb_blk(const uint8_t index);

    void* get_handle_from_mb_blk(MB_BLK blk);
private:
    MB_POOL _mb_pool;
    std::vector<MB_BLK> _ptrs_mb;

    uint16_t _width;
    uint16_t _height;
    uint16_t _bytesPerPixel;
};