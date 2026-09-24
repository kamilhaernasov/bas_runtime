#include "mb_pool.h"

#include <cstring>

mb_pool::mb_pool()
{

}

mb_pool::~mb_pool()
{

}

void mb_pool::init(const uint8_t countBlk, const uint32_t size)
{
    // TODO: проверка на уже выделенный POOL
    _size = size;

    MB_POOL_CONFIG_S mb_pool_cfg;
    memset(&mb_pool_cfg, 0, sizeof(MB_POOL_CONFIG_S));
    mb_pool_cfg.u64MBSize = _size; // Размер одного блока памяти в пуле. Ширина*высота*3 байта на один цвет
	mb_pool_cfg.u32MBCnt = countBlk; // Количество выделяемых блоков в пуле
	mb_pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA; // Тим выделяемой памяти. Надо DMA для аппаратных блоков
	mb_pool_cfg.bPreAlloc = RK_TRUE; // Предварительная аллокация блока вместо ленивого выделения
	_mb_pool = RK_MPI_MB_CreatePool(&mb_pool_cfg);	
}

MB_BLK mb_pool::create_mb_blk(const RK_BOOL block) // блокировать или нет поток
{
    MB_BLK pre_mb = RK_MPI_MB_GetMB(_mb_pool, _size, block); 
    if (pre_mb != nullptr)
    {
        _ptrs_mb.push_back(pre_mb);
    }

    return nullptr;
}

MB_BLK mb_pool::get_mb_blk(const uint8_t index)
{
    if (index < _ptrs_mb.size())
    {
        return (_ptrs_mb[index]);
    }
    return nullptr;
}

void mb_pool::release()
{
    for (auto it = _ptrs_mb.begin(); it != _ptrs_mb.end(); ++it) 
    {
        RK_MPI_MB_ReleaseMB(*it);
    }

    RK_MPI_MB_DestroyPool(_mb_pool);
}

void* mb_pool::get_handle_from_mb_blk(MB_BLK blk)
{
    return RK_MPI_MB_Handle2VirAddr(blk);
}