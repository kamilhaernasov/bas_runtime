#include "mb_pool.h"

#include <cstring>

mb_pool::mb_pool()
{

}

mb_pool::~mb_pool()
{

}

bool mb_pool::init(const uint8_t countBlk, const uint32_t size)
{
    if (_mb_pool != MB_INVALID_POOLID)
    {
        printf("%s: mb_pool already exist!\n", __PRETTY_FUNCTION__);
        return false;
    }
    _size = size;

    MB_POOL_CONFIG_S mb_pool_cfg;
    memset(&mb_pool_cfg, 0, sizeof(MB_POOL_CONFIG_S));
    // Размер одного блока памяти в пуле. Ширина*высота*N байтов на один пиксель
    mb_pool_cfg.u64MBSize = _size; 
    // Количество выделяемых блоков в пуле
	mb_pool_cfg.u32MBCnt = countBlk; 
    // Тим выделяемой памяти. Надо DMA для аппаратных блоков
	mb_pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA; 
    // Предварительная аллокация блока вместо ленивого выделения
	mb_pool_cfg.bPreAlloc = RK_TRUE; 
	_mb_pool = RK_MPI_MB_CreatePool(&mb_pool_cfg);	

    if (_mb_pool == MB_INVALID_POOLID)
    {
        printf("%s: RK_MPI_MB_CreatePool fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    return true;
}

MB_BLK mb_pool::create_mb_blk(const RK_BOOL block) // блокировать или нет поток
{
    MB_BLK pre_mb = RK_MPI_MB_GetMB(_mb_pool, _size, block); 
    if (pre_mb == MB_INVALID_HANDLE)
    {
        printf("%s: RK_MPI_MB_GetMB fail!\n", __PRETTY_FUNCTION__);
        return MB_INVALID_HANDLE;
    }

    _ptrs_mb.push_back(pre_mb);
    return pre_mb;
}

MB_BLK mb_pool::get_mb_blk(const uint8_t index)
{
    if (index < _ptrs_mb.size())
    {
        return (_ptrs_mb[index]);
    }
    printf("%s: get_mb_blk fail! Desc: index not exist\n", __PRETTY_FUNCTION__);
    return MB_INVALID_HANDLE;
}

bool mb_pool::release()
{
    int ret = 0;
    for (auto it = _ptrs_mb.begin(); it != _ptrs_mb.end(); ++it) 
    {
        ret = RK_MPI_MB_ReleaseMB(*it);
        if (ret != 0)
        {
            printf("%s: RK_MPI_MB_ReleaseMB fail! Desc: %d\n", __PRETTY_FUNCTION__, ret);
        }
    }

    ret = RK_MPI_MB_DestroyPool(_mb_pool);
    if (ret != 0)
    {
        printf("%s: RK_MPI_MB_DestroyPool fail! Desc: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

void* mb_pool::get_ptr_from_mb_blk(MB_BLK blk)
{
    return RK_MPI_MB_Handle2VirAddr(blk);
}

bool mb_pool::mmz_flush_cache(const uint8_t index)
{
    int ret = RK_MPI_SYS_MmzFlushCache(
        get_mb_blk(index),
        RK_FALSE
    );

    if (ret != 0)
    {
        printf("%s: RK_MPI_SYS_MmzFlushCache fail! Desc: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}