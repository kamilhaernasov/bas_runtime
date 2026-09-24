#include "vi.h"

vi::vi()
{

}

vi::~vi()
{

}

void vi::set_settings(const settings settings)
{
    _settings = settings;
}

void vi::init()
{
    // ISP INIT
	SAMPLE_COMM_ISP_Init
    (
        _settings._id_camera, // id камеры
        RK_AIQ_WORKING_MODE_NORMAL, // режим hdr
        RK_FALSE, // мультсенсор или нет
        _settings._path_to_iq_dir.c_str()
    );
	SAMPLE_COMM_ISP_Run(_settings._id_camera); // запуск ISP для камеры 0. 
    // Необходимо запускать до VI!
    // please run ISP algorithm to realize automatic exposure control, 
    // automatic gain control, automatic white balance, color correction and other operations to ensure the quality of captured images.

    // MPI INIT
    if (RK_MPI_SYS_Init() != RK_SUCCESS) 
    {
		RK_LOGE("rk mpi sys init fail!");
        return;
	}

    // DEV INIT
    VI_DEV_ATTR_S vi_dev_attr;
	VI_DEV_BIND_PIPE_S vi_dev_bind_pipe;
    VI_CHN_ATTR_S vi_chn_attr;

	memset(&vi_dev_attr, 0, sizeof(VI_DEV_ATTR_S));
	memset(&vi_dev_bind_pipe, 0, sizeof(VI_DEV_BIND_PIPE_S));
	memset(&vi_chn_attr, 0, sizeof(VI_CHN_ATTR_S));

    int ret = 0;
	int devId = 0;
	int pipeId = devId;

	// 0. get dev config status
	ret = RK_MPI_VI_GetDevAttr(devId, &vi_dev_attr);
	if (ret == RK_ERR_VI_NOT_CONFIG) 
    {
		// 0-1.config dev
		ret = RK_MPI_VI_SetDevAttr(devId, &vi_dev_attr);
		if (ret != RK_SUCCESS) 
        {
			printf("RK_MPI_VI_SetDevAttr %x\n", ret);
			return;
		}
	} 
    else 
    {
		printf("RK_MPI_VI_SetDevAttr already\n");
	}
	// 1.get dev enable status
	ret = RK_MPI_VI_GetDevIsEnable(_settings._id_camera);
	if (ret != RK_SUCCESS) 
    {
		// 1-2.enable dev
		ret = RK_MPI_VI_EnableDev(_settings._id_camera);
		if (ret != RK_SUCCESS) 
        {
			printf("RK_MPI_VI_EnableDev %x\n", ret);
			return;
		}
		// 1-3.bind dev/pipe
		vi_dev_bind_pipe.u32Num = 1;
		vi_dev_bind_pipe.PipeId[0] = _settings._id_camera;
		ret = RK_MPI_VI_SetDevBindPipe(devId, &vi_dev_bind_pipe);
		if (ret != RK_SUCCESS) 
        {
			printf("RK_MPI_VI_SetDevBindPipe %x\n", ret);
			return;
		}
	} 
    else 
    {
		printf("RK_MPI_VI_EnableDev already\n");
	}

    // CHN_ID
	vi_chn_attr.stIspOpt.u32BufCount = 2;
	vi_chn_attr.stIspOpt.enMemoryType =
	    VI_V4L2_MEMORY_TYPE_DMABUF; // VI_V4L2_MEMORY_TYPE_MMAP;
	vi_chn_attr.stSize.u32Width = _settings._width;
	vi_chn_attr.stSize.u32Height = _settings._height;
	vi_chn_attr.enPixelFormat = _settings._pixel_format;
	vi_chn_attr.enCompressMode = COMPRESS_MODE_NONE; // COMPRESS_AFBC_16x16;
	vi_chn_attr.u32Depth = 2; //0, get fail, 1 - u32BufCount, can get, if bind to other device, must be < u32BufCount
	ret = RK_MPI_VI_SetChnAttr(0, _settings._id_camera, &vi_chn_attr);
	ret |= RK_MPI_VI_EnableChn(0, _settings._id_camera);
	if (ret) 
    {
		printf("ERROR: create VI error! ret=%d\n", ret);
	}

	_mpp_chn.enModId = RK_ID_VI;
    _mpp_chn.s32DevId = _settings._id_camera;
    _mpp_chn.s32ChnId = 0;
}

void vi::release()
{
	RK_MPI_VI_DisableChn(_settings._id_camera, _settings._id_camera);
	RK_MPI_VI_DisableDev(_settings._id_camera);

	SAMPLE_COMM_ISP_Stop(_settings._id_camera);
}

VIDEO_FRAME_INFO_S* vi::get_frame()
{
    return &_frame;
}

bool vi::receive_frame_from_channel()
{
    RK_S32 result_code = RK_MPI_VI_GetChnFrame(_settings._id_camera, _settings._id_camera, &_frame, -1);
    if (result_code)
    {
        return true;
    }
    return false;
}

void vi::release_frame()
{
	RK_MPI_VI_ReleaseChnFrame(0, 0, &_frame);
}

MPP_CHN_S* vi::get_chn_bind()
{
	return &_mpp_chn;
}