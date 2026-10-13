#include <cstdio>
#include <iostream>
#include "INIReader.h"
#include "br_fsm.h"

// Версия проекта из cmake
#ifndef BAS_RUNTIME_VERSION
#define BAS_RUNTIME_VERSION "unknown"
#endif

// Имена конфигов: рантайм и модель
static constexpr const char* CONFIG_NAME = "config.ini";
static constexpr const char* MODEL_CONFIG_NAME = "model.ini";

bool set_fsm_settings(br_fsm& f);

int main(int argc, char *argv[])
{
    std::cout << "BAS_RUNTIME started. Version: " BAS_RUNTIME_VERSION << std::endl;

    br_fsm bas_fsm;
    if (!set_fsm_settings(bas_fsm))
    {
        printf("%s: %s and %s must be next to the binary\n", __PRETTY_FUNCTION__, CONFIG_NAME, MODEL_CONFIG_NAME);
        return 1;
    }

    if (bas_fsm.init())
    {
        bas_fsm.start();
    }

    bas_fsm.release();

    return 0;
}

bool set_fsm_settings(br_fsm& f)
{
    INIReader config(CONFIG_NAME);
    if (config.ParseError() != 0)
    {
        printf("%s: %s: can't parse config... Error:%s\n", __PRETTY_FUNCTION__, CONFIG_NAME, config.ParseErrorMessage().c_str());
        return false;
    }

    // Настройки модели лежат отдельно: их можно менять, не трогая рантайм
    INIReader model_config(MODEL_CONFIG_NAME);
    if (model_config.ParseError() != 0)
    {
        printf("%s: %s: can't parse config... Error:%s\n", __PRETTY_FUNCTION__, MODEL_CONFIG_NAME, model_config.ParseErrorMessage().c_str());
        return false;
    }

    br_fsm::settings settings;

    // Камера
    settings._width = static_cast<uint16_t>(config.GetUnsigned("camera", "width", settings._width));
    settings._height = static_cast<uint16_t>(config.GetUnsigned("camera", "height", settings._height));
    settings._id_camera = static_cast<uint8_t>(config.GetUnsigned("camera", "id", settings._id_camera));
    settings._path_to_iq_dir = config.GetString("camera", "path_to_iq_dir", settings._path_to_iq_dir);

    // Поток
    settings._codec = config.GetString("venc", "codec", "h264") == "h264" ? br_venc::codec::H264 : settings._codec;
    settings._venc_gop = static_cast<uint16_t>(config.GetUnsigned("venc", "gop", settings._venc_gop));
    settings._venc_bitrate = static_cast<uint32_t>(config.GetUnsigned("venc", "bitrate", settings._venc_bitrate));

    // RTSP
    settings._rtsp_port = static_cast<uint16_t>(config.GetUnsigned("rtsp", "port", settings._rtsp_port));
    settings._rtsp_path = config.GetString("rtsp", "path", settings._rtsp_path);

    // Модель: тип выбирает реализацию разбора, остальное — файл, вход и ручки детекции
    settings._model_type = model_config.GetString("model", "type", settings._model_type);
    settings._path_to_model = model_config.GetString("model", "path", settings._path_to_model);
    settings._width_model = static_cast<uint16_t>(model_config.GetUnsigned("model", "width", settings._width_model));
    settings._height_model = static_cast<uint16_t>(model_config.GetUnsigned("model", "height", settings._height_model));
    settings._path_to_labels = model_config.GetString("model", "labels_path", settings._path_to_labels);
    settings._path_to_anchors = model_config.GetString("model", "anchors_path", settings._path_to_anchors);
    settings._box_thresh = static_cast<float>(model_config.GetReal("detection", "box_thresh", settings._box_thresh));
    settings._nms_thresh = static_cast<float>(model_config.GetReal("detection", "nms_thresh", settings._nms_thresh));
    settings._max_detections = static_cast<int>(model_config.GetUnsigned("detection", "max_detections", settings._max_detections));

    f.set_settings(settings);

    printf("%s: %s: apply: | frame %ux%u | rtsp %u%s\n",
           __PRETTY_FUNCTION__, CONFIG_NAME,
           static_cast<unsigned>(settings._width), static_cast<unsigned>(settings._height),
           static_cast<unsigned>(settings._rtsp_port), settings._rtsp_path.c_str());
    printf("%s: %s: apply: | type %s | model %s | input %ux%u | box %.2f | nms %.2f | max %d\n",
           __PRETTY_FUNCTION__, MODEL_CONFIG_NAME,
           settings._model_type.c_str(), settings._path_to_model.c_str(),
           static_cast<unsigned>(settings._width_model), static_cast<unsigned>(settings._height_model),
           settings._box_thresh, settings._nms_thresh, settings._max_detections);

    return true;
}
