#include "fsm.h"

#include <iostream>

RK_U64 TEST_COMM_GetNowUs() {
	struct timespec time = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &time);
	return (RK_U64)time.tv_sec * 1000000 + (RK_U64)time.tv_nsec / 1000; /* microseconds */
}

fsm::fsm()
{

}

fsm::~fsm()
{

}

bool fsm::init()
{
    create_mb_pool(1);
    MB_BLK src_Blk = RK_MPI_MB_GetMB(_mb_pool, _width * _height * 3, RK_TRUE);
    build_h264_frame(&src_Blk);
    isp_init();
    mpi_init();
    rtsp_init();
    vi_dev_init();
    vi_channel_init();
    venc_init();

	VENC_STREAM_S stFrame;	
	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
	RK_U64 H264_PTS = 0;
	RK_U32 H264_TimeRef = 0; 
	VIDEO_FRAME_INFO_S stViFrame;

    RK_S32 s32Ret = 0;
    float fps;

    while(1) {			
		// get vi frame
        _h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
		_h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs(); 
		s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
		//std::cout << "frame receiverd" << std::endl;
		// encode H264	
		RK_MPI_VENC_SendFrame(0,  &stViFrame ,-1);
        //std::cout << "frame send" << std::endl;
		// rtsp
		s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);	
		if(s32Ret == RK_SUCCESS) {
			if(_rtsp_handle && _rtsp_session) {
				//printf("len = %d PTS = %d \n",stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);	
				void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
				rtsp_tx_video(_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
							  stFrame.pstPack->u64PTS);
				rtsp_do_event(_rtsp_handle);
			}
			RK_U64 nowUs = TEST_COMM_GetNowUs();
			fps = (float) 1000000 / (float)(nowUs - _h264_frame.stVFrame.u64PTS);	
            std::cout << fps << std::endl;		
		}

		// release frame 
		s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
		}
		s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
		}
	
	}

	// Destory MB
	RK_MPI_MB_ReleaseMB(src_Blk);
	// Destory Pool
	RK_MPI_MB_DestroyPool(_mb_pool);

	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);
		
	SAMPLE_COMM_ISP_Stop(0);

	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);

	if (_rtsp_handle)
		rtsp_del_demo(_rtsp_handle);
	
	RK_MPI_SYS_Exit();

    return true;
}

void fsm::set_width(const uint16_t width)
{
   _width = width;
}

void fsm::set_height(const uint16_t height)
{
    _height = height;
}

uint16_t fsm::get_width() const
{
    return _width;
}

uint16_t fsm::get_height() const
{
    return _height;
}

bool fsm::create_mb_pool(const uint8_t mb_cnt)
{
    memset(&_mb_pool_cfg, 0, sizeof(MB_POOL_CONFIG_S));
    _mb_pool_cfg.u64MBSize = _width * _height * 3;      // Размер одного блока памяти в пуле. Ширина*высота*3 байта на один цвет
	_mb_pool_cfg.u32MBCnt = mb_cnt;                     // Количество выделяемых блоков в пуле
	_mb_pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;       // Тим выделяемой памяти. Надо DMA для аппаратных блоков
	_mb_pool_cfg.bPreAlloc = RK_TRUE;                   // Предварительная аллокация блока вместо ленивого выделения
	_mb_pool = RK_MPI_MB_CreatePool(&_mb_pool_cfg);	

    return true;
}

bool fsm::build_h264_frame(MB_BLK* mb_blk)
{
    _h264_frame.stVFrame.u32Width = _width;
	_h264_frame.stVFrame.u32Height = _height;
	_h264_frame.stVFrame.u32VirWidth = _width;
	_h264_frame.stVFrame.u32VirHeight = _height;
	_h264_frame.stVFrame.enPixelFormat =  RK_FMT_YUV420SP; // формат кодирования для VENC
	_h264_frame.stVFrame.u32FrameFlag = 160; // todo: разобраться
	_h264_frame.stVFrame.pMbBlk = mb_blk; // указатель на блок из MB POOL. VENC будет читать из этого блока

    return true;
}

bool fsm::isp_init()
{
	const char *iq_dir = "/etc/iqfiles"; // Каталог с файлами калибровки ISP
	rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL; // режим hdr

	SAMPLE_COMM_ISP_Init(
        0, // id камеры
        hdr_mode, // режим hdr
        RK_FALSE, // мультсенсор или нет
        iq_dir
    );
	SAMPLE_COMM_ISP_Run(0); // запуск ISP для камеры 0. 
    // Необходимо запускать до VI!
    // please run ISP algorithm to realize automatic exposure control, 
    // automatic gain control, automatic white balance, color correction and other operations to ensure the quality of captured images.

    return true;
}

bool fsm::mpi_init()
{
    // MPI система
	if (RK_MPI_SYS_Init() != RK_SUCCESS) 
    {
		RK_LOGE("rk mpi sys init fail!");
		return false;
	}
    return true;
}

bool fsm::rtsp_init()
{
    _rtsp_handle = NULL;
	_rtsp_handle = create_rtsp_demo(554); // создание сервера на порту 554
	_rtsp_session = rtsp_new_session(_rtsp_handle, "/live/0"); // путь до стрима
	rtsp_set_video(_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0); // видео в h264
	rtsp_sync_video_ts(_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime()); // синхронизация времени с ntp
}

bool fsm::vi_dev_init()
{
    int ret = 0;
	int devId = 0;
	int pipeId = devId;

	memset(&_vi_dev_attr, 0, sizeof(_vi_dev_attr));
	memset(&_vi_dev_bind_pipe, 0, sizeof(_vi_dev_bind_pipe));
	// 0. get dev config status
	ret = RK_MPI_VI_GetDevAttr(devId, &_vi_dev_attr);
	if (ret == RK_ERR_VI_NOT_CONFIG) {
		// 0-1.config dev
		ret = RK_MPI_VI_SetDevAttr(devId, &_vi_dev_attr);
		if (ret != RK_SUCCESS) {
			printf("RK_MPI_VI_SetDevAttr %x\n", ret);
			return -1;
		}
	} else {
		printf("RK_MPI_VI_SetDevAttr already\n");
	}
	// 1.get dev enable status
	ret = RK_MPI_VI_GetDevIsEnable(devId);
	if (ret != RK_SUCCESS) {
		// 1-2.enable dev
		ret = RK_MPI_VI_EnableDev(devId);
		if (ret != RK_SUCCESS) {
			printf("RK_MPI_VI_EnableDev %x\n", ret);
			return -1;
		}
		// 1-3.bind dev/pipe
		_vi_dev_bind_pipe.u32Num = 1;
		_vi_dev_bind_pipe.PipeId[0] = pipeId;
		ret = RK_MPI_VI_SetDevBindPipe(devId, &_vi_dev_bind_pipe);
		if (ret != RK_SUCCESS) {
			printf("RK_MPI_VI_SetDevBindPipe %x\n", ret);
			return -1;
		}
	} else {
		printf("RK_MPI_VI_EnableDev already\n");
	}

	return 0;
}

bool fsm::vi_channel_init()
{
    int ch_id = 0;
    int ret;
	int buf_cnt = 2;
	// VI init
	memset(&_vi_chn_attr, 0, sizeof(_vi_chn_attr));
	_vi_chn_attr.stIspOpt.u32BufCount = buf_cnt;
	_vi_chn_attr.stIspOpt.enMemoryType =
	    VI_V4L2_MEMORY_TYPE_DMABUF; // VI_V4L2_MEMORY_TYPE_MMAP;
	_vi_chn_attr.stSize.u32Width = _width;
	_vi_chn_attr.stSize.u32Height = _height;
	_vi_chn_attr.enPixelFormat = RK_FMT_YUV420SP;
	_vi_chn_attr.enCompressMode = COMPRESS_MODE_NONE; // COMPRESS_AFBC_16x16;
	_vi_chn_attr.u32Depth = 2; //0, get fail, 1 - u32BufCount, can get, if bind to other device, must be < u32BufCount
	ret = RK_MPI_VI_SetChnAttr(0, ch_id, &_vi_chn_attr);
	ret |= RK_MPI_VI_EnableChn(0, ch_id);
	if (ret) {
		printf("ERROR: create VI error! ret=%d\n", ret);
		return ret;
	}

	return ret;
}

bool fsm::venc_init()
{
    printf("%s\n",__func__);

	memset(&_venc_chn_attr, 0, sizeof(VENC_CHN_ATTR_S));
	RK_CODEC_ID_E enType = RK_VIDEO_ID_AVC;
	if (enType == RK_VIDEO_ID_AVC) {
		_venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
		_venc_chn_attr.stRcAttr.stH264Cbr.u32BitRate = 10 * 1024;
		_venc_chn_attr.stRcAttr.stH264Cbr.u32Gop = 1;
	} else if (enType == RK_VIDEO_ID_HEVC) {
		_venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
		_venc_chn_attr.stRcAttr.stH265Cbr.u32BitRate = 10 * 1024;
		_venc_chn_attr.stRcAttr.stH265Cbr.u32Gop = 60;
	} else if (enType == RK_VIDEO_ID_MJPEG) {
		_venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_MJPEGCBR;
		_venc_chn_attr.stRcAttr.stMjpegCbr.u32BitRate = 10 * 1024;
	}

	_venc_chn_attr.stVencAttr.enType = enType;
	_venc_chn_attr.stVencAttr.enPixelFormat = RK_FMT_YUV420SP;
	if (enType == RK_VIDEO_ID_AVC)
		_venc_chn_attr.stVencAttr.u32Profile = H264E_PROFILE_HIGH;
	_venc_chn_attr.stVencAttr.u32PicWidth = _width;
	_venc_chn_attr.stVencAttr.u32PicHeight = _height;
	_venc_chn_attr.stVencAttr.u32VirWidth = _width;
	_venc_chn_attr.stVencAttr.u32VirHeight = _height;
	_venc_chn_attr.stVencAttr.u32StreamBufCnt = 2;
	_venc_chn_attr.stVencAttr.u32BufSize = _width * _height * 3 / 2;
	_venc_chn_attr.stVencAttr.enMirror = MIRROR_NONE;

	RK_MPI_VENC_CreateChn(0, &_venc_chn_attr);

	memset(&_venc_recv_param, 0, sizeof(VENC_RECV_PIC_PARAM_S));
	_venc_recv_param.s32RecvPicNum = -1;
	RK_MPI_VENC_StartRecvFrame(0, &_venc_recv_param);

	return 0;
}