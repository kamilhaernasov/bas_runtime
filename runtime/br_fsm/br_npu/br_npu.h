#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C"
{
    #include "rk_comm_mb.h"
    #include "rk_mpi_mb.h"
    #include "rk_mpi_vi.h"
    #include "rk_comm_video.h"
    #include "rk_aiq.h"
    #include "sample_comm.h"
    #include "rtsp_demo.h"
};

// yolov5.h тянет за собой rknn_api.h и postprocess.h, а postprocess.h включает <vector>,
// поэтому его нельзя класть в extern "C". Отсюда приходят:
// rknn_app_context_t, object_detect_result_list и константы постобработки (OBJ_CLASS_NUM, BOX_THRESH, NMS_THRESH и т.д.)
#include "yolov5.h"

class br_npu
{
public:
    struct settings
    {
        std::string _model_path;
        std::string _label_path;
    };

    explicit br_npu();
    ~br_npu();

    bool init();
    void set_settings(const settings settings);

    // На вход подается уже подготовленный кадр (letterbox) размером model_width*model_height*3
    bool inference(const uint8_t* ptr_model_input, object_detect_result_list* ptr_od_results);

    bool release();

    int get_model_width() const;
    int get_model_height() const;
    const char* class_name(int cls_id) const;
private:
    void load_labels();
    void post_process(object_detect_result_list* ptr_od_results);

    settings _settings;

    rknn_context _ctx = 0;
    rknn_app_context_t _app_ctx {};

    int _model_width = 0;
    int _model_height = 0;

    std::vector<std::string> _labels;
};