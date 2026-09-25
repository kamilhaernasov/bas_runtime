#include "rtsp.h"

rtsp::rtsp()
{

}

rtsp::~rtsp()
{

}

bool rtsp::init()
{
    int ret = 0;
    // создание сервера на порту 554
	_rtsp_handle = create_rtsp_demo(_settings._port); 
    if (_rtsp_handle == nullptr)
    {
        printf("%s: create_rtsp_demo fail!\n", __PRETTY_FUNCTION__);
		return false;
    }
	_rtsp_session = rtsp_new_session(_rtsp_handle, _settings._path.c_str()); 
    if (_rtsp_session == nullptr)
    {
        printf("%s: rtsp_new_session fail!\n", __PRETTY_FUNCTION__);
		return false;
    }

	ret = rtsp_set_video(_rtsp_session, _settings._codec_rtsp, NULL, 0); // видео в h264
    if (ret < 0)
    {
        printf("%s: rtsp_set_video fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
    }

	ret = rtsp_sync_video_ts(_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime()); // синхронизация времени с ntp
    if (ret < 0)
    {
        printf("%s: rtsp_sync_video_ts fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
    }

    return true;
}

void rtsp::release()
{
	if (_rtsp_handle)
    {
		rtsp_del_demo(_rtsp_handle);
    }
}

void rtsp::set_settings(const settings settings)
{
    _settings = settings;
}

bool rtsp::process_frame(uint8_t* ptr_data, const uint32_t len_data, const uint64_t pts_data)
{
    int ret = 0;

    ret = rtsp_tx_video(_rtsp_session, ptr_data, len_data, pts_data);
    if (ret < 0)
    {
        printf("%s: rtsp_tx_video fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
    }

    ret = rtsp_do_event(_rtsp_handle);
    if (ret < 0)
    {
        printf("%s: rtsp_do_event fail! ret=%x\n", __PRETTY_FUNCTION__, ret);
		return false;
    }

    return true;
}