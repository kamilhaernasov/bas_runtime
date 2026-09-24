#include "vpss.h"

vpss::vpss()
{

}

vpss::~vpss()
{

}

bool vpss::init()
{
    // GROUP INIT
    VPSS_GRP_ATTR_S vpss_grp_attr;
    memset(&vpss_grp_attr, 0, sizeof(vpss_grp_attr));
    vpss_grp_attr.u32MaxW = _settings._width;
    vpss_grp_attr.u32MaxH = _settings._height;
    vpss_grp_attr.enPixelFormat = _settings._format_input;  
    vpss_grp_attr.stFrameRate.s32SrcFrameRate = -1;
    vpss_grp_attr.stFrameRate.s32DstFrameRate = -1;
    vpss_grp_attr.enCompressMode = COMPRESS_MODE_NONE;

    RK_MPI_VPSS_CreateGrp(0, &vpss_grp_attr);
    RK_MPI_VPSS_StartGrp(0);

    // CHN INIT
    VPSS_CHN_ATTR_S vpss_chn_attr;
    memset(&vpss_chn_attr, 0, sizeof(vpss_chn_attr));
    vpss_chn_attr.enChnMode = VPSS_CHN_MODE_USER;
    vpss_chn_attr.enPixelFormat = _settings._format_output;   // выход: RGB888
    vpss_chn_attr.enCompressMode = COMPRESS_MODE_NONE;
    vpss_chn_attr.u32Width = _settings._width;
    vpss_chn_attr.u32Height = _settings._height;
    vpss_chn_attr.stFrameRate.s32SrcFrameRate = -1;
    vpss_chn_attr.stFrameRate.s32DstFrameRate = -1;

    RK_MPI_VPSS_SetChnAttr(0, 0, &vpss_chn_attr);
    RK_MPI_VPSS_EnableChn(0, 0);

    _mpp_chn.enModId = RK_ID_VPSS;
    _mpp_chn.s32DevId = 0;
    _mpp_chn.s32ChnId = 0;
}

MPP_CHN_S* vpss::get_chn_bind()
{
    return &_mpp_chn;
}

void vpss::set_settings(const settings settings)
{
    _settings = settings;
}

bool vpss::receive_frame()
{
    int ret = RK_MPI_VPSS_GetChnFrame(0, 0, &_frame, 2000);
    if (ret != RK_SUCCESS) {
        printf("VPSS GetChnFrame failed: 0x%x\n", ret);
    }
}

VIDEO_FRAME_INFO_S* vpss::get_frame()
{
    return &_frame;
}

bool vpss::release_frame()
{
    RK_MPI_VPSS_ReleaseChnFrame(0, 0, &_frame);
}