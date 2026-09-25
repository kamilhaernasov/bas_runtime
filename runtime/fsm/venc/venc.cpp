#include "venc.h"

uint64_t venc::get_current_time_us() const
{
    static struct timespec time = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000000 + time.tv_nsec / 1000;
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

bool venc::init(const uint16_t channel, MB_BLK ptr_blk)
{
    int ret = 0;

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
    memset(&venc_chn_attr, 0, sizeof(VENC_CHN_ATTR_S));

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

    VENC_RECV_PIC_PARAM_S venc_pic_param;
	memset(&venc_pic_param, 0, sizeof(VENC_RECV_PIC_PARAM_S));
    venc_pic_param.s32RecvPicNum = -1;

    ret = RK_MPI_VENC_CreateChn(_channel, &venc_chn_attr);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_CreateChn fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    ret = RK_MPI_VENC_StartRecvFrame(_channel, &venc_pic_param);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_StartRecvFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    return true;
}

bool venc::release()
{
    int ret = 0;

    ret = RK_MPI_VENC_StopRecvFrame(_channel);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_StopRecvFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	ret = RK_MPI_VENC_DestroyChn(_channel);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_DestroyChn fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	free(_codec_frame.pstPack);

    return true;
}

bool venc::prepare_frame(VIDEO_FRAME_INFO_S* frame)
{
    int ret = 0;
    static RK_U32 H264_TimeRef = 0; 

    _frame.stVFrame.u32TimeRef = H264_TimeRef++;
    _frame.stVFrame.u64PTS = get_current_time_us(); 
    
    ret = RK_MPI_VENC_SendFrame(_channel, &_frame ,-1);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_SendFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    return true;
}

bool venc::process_frame()
{
    int ret = 0;

    ret = RK_MPI_VENC_GetStream(_channel, &_codec_frame, -1);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_SendFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}
    return true;
}

VENC_STREAM_S* venc::get_codec_frame()
{
    return &_codec_frame;
}

bool venc::release_codec_frame()
{
    int ret = 0;

    ret = RK_MPI_VENC_ReleaseStream(_channel, &_codec_frame);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VENC_ReleaseStream fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    return true;
}