#include "venc.h"

RK_U64 TEST_COMM_GetNowUs() {
	struct timespec time = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &time);
	return (RK_U64)time.tv_sec * 1000000 + (RK_U64)time.tv_nsec / 1000; /* microseconds */
}

venc::venc()
{

}

venc::~venc()
{

}

void venc::set_settings(const settings settings)
{
    _settings = settings;
}

void venc::init(const uint16_t channel, MB_BLK ptr_blk)
{
    _codec_frame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));

	_frame.stVFrame.u32Width = _settings._width;
	_frame.stVFrame.u32Height = _settings._height;
	_frame.stVFrame.u32VirWidth = _settings._width;
	_frame.stVFrame.u32VirHeight = _settings._height;
	_frame.stVFrame.enPixelFormat =  _settings._pixel_format; 
	_frame.stVFrame.u32FrameFlag = 160;
	_frame.stVFrame.pMbBlk = ptr_blk;

    _channel = channel;

    VENC_CHN_ATTR_S venc_chn_attr;
    VENC_RECV_PIC_PARAM_S venc_pic_param;

    memset(&venc_chn_attr, 0, sizeof(VENC_CHN_ATTR_S));
	memset(&venc_pic_param, 0, sizeof(VENC_RECV_PIC_PARAM_S));

    switch (_settings._codec)
    {
    case codec::H264:
        venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
        venc_chn_attr.stRcAttr.stH264Cbr.u32BitRate = _settings._bitrate;
        venc_chn_attr.stRcAttr.stH264Cbr.u32Gop = _settings._gop;
        venc_chn_attr.stVencAttr.enType = RK_VIDEO_ID_AVC;
        venc_chn_attr.stVencAttr.enPixelFormat = _settings._pixel_format;
        venc_chn_attr.stVencAttr.u32Profile = H264E_PROFILE_HIGH;
        venc_chn_attr.stVencAttr.u32PicWidth = _settings._width;
        venc_chn_attr.stVencAttr.u32PicHeight = _settings._height;
        venc_chn_attr.stVencAttr.u32VirWidth = _settings._width;
        venc_chn_attr.stVencAttr.u32VirHeight = _settings._height;
        venc_chn_attr.stVencAttr.u32StreamBufCnt = 2;
        venc_chn_attr.stVencAttr.u32BufSize = _settings._width * _settings._height * _settings._bytes_per_pixels / 2;
        venc_chn_attr.stVencAttr.enMirror = MIRROR_NONE;
        break;
    }
	memset(&venc_pic_param, 0, sizeof(VENC_RECV_PIC_PARAM_S));
    venc_pic_param.s32RecvPicNum = -1;

    RK_MPI_VENC_CreateChn(_channel, &venc_chn_attr);
    RK_MPI_VENC_StartRecvFrame(_channel, &venc_pic_param);
}

void venc::release()
{
    RK_MPI_VENC_StopRecvFrame(_channel);
	RK_MPI_VENC_DestroyChn(_channel);
	free(_codec_frame.pstPack);
}

void venc::exec_frame_from_vi(VIDEO_FRAME_INFO_S* frame)
{
    static RK_U32 H264_TimeRef = 0; 

    _frame.stVFrame.u32TimeRef = H264_TimeRef++;
    _frame.stVFrame.u64PTS = TEST_COMM_GetNowUs(); 
    RK_MPI_VENC_SendFrame(0, frame ,-1);
}

void venc::exec_frame_to_codec()
{
    RK_MPI_VENC_GetStream(0, &_codec_frame, -1);
}

VENC_STREAM_S* venc::get_codec_frame()
{
    return &_codec_frame;
}

void venc::release_frame()
{
    RK_MPI_VENC_ReleaseStream(0, &_codec_frame);
}