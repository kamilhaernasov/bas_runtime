#include "br_npu.h"

br_npu::br_npu()
{

}

br_npu::~br_npu()
{

}

bool br_npu::init()
{
    int ret = 0;
    ret = rknn_init(&_ctx, _settings._model_path.c_str(), 0, 0, nullptr);
    if (ret < 0) 
    {
        printf("%s: rknn_init fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(&_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("%s: rknn_query fail! Code: %d\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

    return true;
}

void br_npu::set_settings(const settings settings)
{
    _settings = settings;
}

