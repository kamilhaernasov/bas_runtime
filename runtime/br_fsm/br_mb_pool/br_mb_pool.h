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

class br_mb_pool
{
public:
    explicit br_mb_pool();
    ~br_mb_pool();

    bool init(const uint8_t countBlk, const uint32_t size);
    bool release();

    MB_BLK create_mb_blk(const RK_BOOL block);
    MB_BLK get_mb_blk(const uint8_t index);

    void* get_ptr_from_mb_blk(MB_BLK blk);

    bool mmz_flush_cache(const uint8_t index);
    // Сброс кэша в другую сторону: буфер писала железка, а читает его процессор.
    // Без этого процессор может прочитать то, что осталось в кэше с прошлого раза
    bool mmz_invalidate_cache(const uint8_t index);
private:
    MB_POOL _mb_pool = MB_INVALID_POOLID;
    std::vector<MB_BLK> _ptrs_mb;

    uint32_t _size;
};