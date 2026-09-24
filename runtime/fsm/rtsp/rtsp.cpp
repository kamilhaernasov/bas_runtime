#include "rtsp.h"

rtsp::rtsp()
{

}

rtsp::~rtsp()
{

}

void rtsp::init()
{
    _rtsp_handle = NULL;
	_rtsp_handle = create_rtsp_demo(_settings._port); // создание сервера на порту 554
	_rtsp_session = rtsp_new_session(_rtsp_handle, _settings._path.c_str()); // путь до стрима
	rtsp_set_video(_rtsp_session, _settings._codec_rtsp, NULL, 0); // видео в h264
	rtsp_sync_video_ts(_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime()); // синхронизация времени с ntp
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

void rtsp::send_frame(uint8_t* ptr_data, const uint32_t len_data, const uint64_t pts_data)
{
    rtsp_tx_video(_rtsp_session, ptr_data, len_data, pts_data);
    rtsp_do_event(_rtsp_handle);
}