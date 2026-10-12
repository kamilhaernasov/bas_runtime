#include <cstdio>
#include <iostream>
#include "INIReader.h"
#include "br_fsm.h"

bool set_fsm_settings(br_fsm& f, const std::string& config_name);

int main(int argc, char *argv[])
{
    std::cout << "Start..." << std::endl;

    const std::string config_name = argc > 1 ? argv[1] : "config.ini";

    br_fsm bas_fsm;
    if (!set_fsm_settings(bas_fsm, config_name))
    {
        printf("%s: %s fail!\n", __PRETTY_FUNCTION__, config_name.c_str());
        return 1;
    }

    if (bas_fsm.init())
    {
        bas_fsm.start();
    }

    bas_fsm.release();

    return 0;
}

bool set_fsm_settings(br_fsm& f, const std::string& config_name)
{
    INIReader reader(config_name);

    if (reader.ParseError() != 0)
    {
        printf("%s: %s: %s\n", __PRETTY_FUNCTION__, config_name.c_str(), reader.ParseErrorMessage().c_str());
        return false;
    }

    // В settings уже лежат значения по умолчанию: они и подставляются для
    // отсутствующего ключа или значения, которое не удалось разобрать
    br_fsm::settings settings;

    // Камера
    settings._width = static_cast<uint16_t>(reader.GetUnsigned("camera", "width", settings._width));
    settings._height = static_cast<uint16_t>(reader.GetUnsigned("camera", "height", settings._height));
    settings._id_camera = static_cast<uint8_t>(reader.GetUnsigned("camera", "id", settings._id_camera));
    settings._path_to_iq_dir = reader.GetString("camera", "path_to_iq_dir", settings._path_to_iq_dir);

    // Модель
    settings._width_model = static_cast<uint16_t>(reader.GetUnsigned("model", "width", settings._width_model));
    settings._height_model = static_cast<uint16_t>(reader.GetUnsigned("model", "height", settings._height_model));
    settings._path_to_model = reader.GetString("model", "path", settings._path_to_model);
    settings._path_to_labels = reader.GetString("model", "labels_path", settings._path_to_labels);
    settings._path_to_anchors = reader.GetString("model", "anchors_path", settings._path_to_anchors);
    settings._box_thresh = static_cast<float>(reader.GetReal("model", "box_thresh", settings._box_thresh));
    settings._nms_thresh = static_cast<float>(reader.GetReal("model", "nms_thresh", settings._nms_thresh));

    // Поток. Кодек пока один, поэтому все, кроме h264, оставляет дефолт
    settings._codec = reader.GetString("venc", "codec", "h264") == "h264" ? br_venc::codec::H264 : settings._codec;
    settings._venc_gop = static_cast<uint16_t>(reader.GetUnsigned("venc", "gop", settings._venc_gop));
    settings._venc_bitrate = static_cast<uint32_t>(reader.GetUnsigned("venc", "bitrate", settings._venc_bitrate));
    settings._rtsp_port = static_cast<uint16_t>(reader.GetUnsigned("rtsp", "port", settings._rtsp_port));
    settings._rtsp_path = reader.GetString("rtsp", "path", settings._rtsp_path);

    f.set_settings(settings);

    printf("%s: %s: ok | frame %ux%u | model %ux%u %s | rtsp %u%s\n",
           __PRETTY_FUNCTION__, config_name.c_str(),
           static_cast<unsigned>(settings._width), static_cast<unsigned>(settings._height),
           static_cast<unsigned>(settings._width_model), static_cast<unsigned>(settings._height_model),
           settings._path_to_model.c_str(),
           static_cast<unsigned>(settings._rtsp_port), settings._rtsp_path.c_str());

    return true;
}