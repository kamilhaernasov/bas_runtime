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

bool vi::init()
{
	int32_t ret = 0;

    // ISP INIT
	ret = SAMPLE_COMM_ISP_Init
    (
		// id камеры
        _settings._id_camera, 
		// режим hdr
        RK_AIQ_WORKING_MODE_NORMAL, 
		 // мультсенсор или нет
        RK_FALSE, 
		// путь до папки с ISP конфигами
        _settings._path_to_iq_dir.c_str()
    );
	if (ret != RK_SUCCESS)
    {
        printf("%s: SAMPLE_COMM_ISP_Init fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
        return false;
    }

	// ISP RUN
	// запуск ISP для камеры 0. 
	// Необходимо запускать до VI!
    // please run ISP algorithm to realize automatic exposure control, 
    // automatic gain control, automatic white balance, color correction and other operations to ensure the quality of captured images.
	ret = SAMPLE_COMM_ISP_Run(_settings._id_camera); 
	if (ret != RK_SUCCESS)
	{
		printf("%s: SAMPLE_COMM_ISP_Run fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    // MPI INIT
	ret = RK_MPI_SYS_Init();
    if (ret != RK_SUCCESS) 
    {
		printf("%s: RK_MPI_SYS_Init fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
        return false;
	}

    // DEV INIT
    VI_DEV_ATTR_S vi_dev_attr;
	memset(&vi_dev_attr, 0, sizeof(VI_DEV_ATTR_S));
	// Получаем атрибуты устройства vi
	ret = RK_MPI_VI_GetDevAttr(_dev_id, &vi_dev_attr);
	if (ret != RK_ERR_VI_NOT_CONFIG)
	{
		printf("%s: RK_MPI_VI_GetDevAttr fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
        return false;
	}
	// Пока никаких атрибутов не ставим, но на будущее
	// Устанавливаем атрибуты vi устройства
	ret = RK_MPI_VI_SetDevAttr(_dev_id, &vi_dev_attr);
	if (ret != RK_SUCCESS) 
	{
		printf("%s: RK_MPI_VI_SetDevAttr fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	// Узнаем, запущено ли устройство vi
	ret = RK_MPI_VI_GetDevIsEnable(_settings._id_camera);
	if (ret == RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_GetDevIsEnable fail! Already run! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	// Включаем устройство
	ret = RK_MPI_VI_EnableDev(_settings._id_camera);
	if (ret != RK_SUCCESS) 
	{
		printf("%s: RK_MPI_VI_EnableDev fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	// PIPE INIT
	VI_DEV_BIND_PIPE_S vi_dev_bind_pipe;
	memset(&vi_dev_bind_pipe, 0, sizeof(VI_DEV_BIND_PIPE_S));
	// Связываем с pipe
	vi_dev_bind_pipe.u32Num = 1;
	vi_dev_bind_pipe.PipeId[0] = _pipe_id;
	ret = RK_MPI_VI_SetDevBindPipe(_dev_id, &vi_dev_bind_pipe);
	if (ret != RK_SUCCESS) 
	{
		printf("%s: RK_MPI_VI_SetDevBindPipe fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	// CHANNEL INIT
 	VI_CHN_ATTR_S vi_chn_attr;
	memset(&vi_chn_attr, 0, sizeof(VI_CHN_ATTR_S));
	// Количество буферов, выделяемых для ISP (двойная буферизация: пока один кадр обрабатывается, второй заполняется)
	vi_chn_attr.stIspOpt.u32BufCount = 2;
	// Тип памяти буферов — DMABUF (zero-copy: буферы передаются между драйвером и приложением без копирования)
	vi_chn_attr.stIspOpt.enMemoryType = VI_V4L2_MEMORY_TYPE_DMABUF; 
	 // Ширина кадра, который будет выдавать канал vi
	vi_chn_attr.stSize.u32Width = _settings._width;
	 // Высота кадра, который будет выдавать канал vi
	vi_chn_attr.stSize.u32Height = _settings._height;
	// Формат пикселей (например NV12)
	vi_chn_attr.enPixelFormat = _settings._pixel_format;
	// Режим компрессии кадра — без сжатия (сырой несжатый кадр)
	vi_chn_attr.enCompressMode = COMPRESS_MODE_NONE; 
	 // Глубина очереди каналов: сколько кадров может накопиться в очереди на выдачу пользователю
	vi_chn_attr.u32Depth = 2; 
	ret = RK_MPI_VI_SetChnAttr(_pipe_id, _chn_id, &vi_chn_attr);
	if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_SetChnAttr fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	ret = RK_MPI_VI_EnableChn(_pipe_id, _chn_id);
	if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_SetChnAttr fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	return true;
}

bool vi::release()
{
	int ret = 0;

	ret = RK_MPI_VI_DisableChn(_pipe_id, _chn_id);
	if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_DisableChn fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	ret = RK_MPI_VI_DisableDev(_dev_id);
	if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_DisableDev fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	ret = SAMPLE_COMM_ISP_Stop(_settings._id_camera);
	if (ret != RK_SUCCESS)
	{
		printf("%s: SAMPLE_COMM_ISP_Stop fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	return true;
}

VIDEO_FRAME_INFO_S* vi::get_last_frame()
{
    return &_frame;
}

bool vi::receive_frame_from_channel()
{
	// -1 - режим блокировки, ждем кадр
    int ret = RK_MPI_VI_GetChnFrame(_pipe_id, _chn_id, &_frame, -1);
    if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_GetChnFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

    return true;
}

bool vi::release_frame()
{
	int ret = RK_MPI_VI_ReleaseChnFrame(_pipe_id, _chn_id, &_frame);
	if (ret != RK_SUCCESS)
	{
		printf("%s: RK_MPI_VI_ReleaseChnFrame fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
	}

	return true;
}