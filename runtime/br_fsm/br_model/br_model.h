#pragma once

/**
 * @file br_model.h
 * @author khaernasov-ka
 * @brief Интерфейс разбора выходов модели.
 * @version 0.1
 * @date 2026-10-13
 *
 * br_npu владеет RKNN: загружает .rknn, привязывает память тензоров (zero-copy) и гоняет
 * модель в отдельном потоке. Как устроен выход конкретной модели — сколько голов, какие
 * якоря, где уверенность объекта и оценки классов — знает реализация этого интерфейса.
 * Реализация создается по типу модели из конфига (br_model_create).
 *
 * Обмен идет в двух точках:
 * ├── init    - br_npu отдает форму модели, реализация проверяет ее и готовит разбор
 * └── process - br_npu отдает выходные тензоры прогона, реализация возвращает рамки
 *               в координатах кадра
 *
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class br_model
{
public:
    // Прямоугольник в координатах кадра.
    // left, top — левый верхний угол.
    // right, bottom — правый нижний угол.
    struct rect
    {
        int left;
        int top;
        int right;
        int bottom;
    };

    // Рамка детекции в координатах кадра — то, что уходит на отрисовку.
    // box — прямоугольник рамки.
    // prop — уверенность детекции (0..1).
    // cls_id — номер класса модели, по нему br_fsm берет название через get_class_name.
    struct detection
    {
        rect box;
        float prop;
        int cls_id;
    };

    // Список рамок одного кадра: порядок — по убыванию уверенности
    typedef std::vector<detection> detection_list;

    // Настройки модели: файл, ее вход и ручки разбора плюс геометрия кадра.
    // _path_to_model — файл модели: br_npu отдает его в rknn_init.
    // _width_model, _height_model — размер входа модели: под него br_fsm выделяет блок
    //                               кадра, в который кадр укладывает RGA.
    // _box_thresh — порог уверенности детекции: слабее него рамки не берутся.
    // _nms_thresh — порог подавления дубликатов: если две рамки одного класса пересекаются
    //               сильнее, слабая выбрасывается.
    // _max_detections — сколько рамок помещается в кадр.
    // _path_to_labels — файл с названиями классов модели.
    // _path_to_anchors — файл с якорями голов модели.
    // _width, _height — размер кадра.
    // _scale, _pad_x, _pad_y — геометрия вписывания кадра во вход модели (считает RGA):
    //                          по ним рамки возвращаются из пространства модели в кадр.
    struct settings
    {
        std::string _path_to_model;
        uint16_t _width_model = 0;
        uint16_t _height_model = 0;
        float _box_thresh = 0;
        float _nms_thresh = 0;
        int _max_detections = 0;
        std::string _path_to_labels;
        std::string _path_to_anchors;
        uint16_t _width = 0;
        uint16_t _height = 0;
        float _scale = 0;
        uint16_t _pad_x = 0;
        uint16_t _pad_y = 0;
    };

    // Один выходной тензор прогона: то, что br_npu привязал к модели.
    // _data — данные тензора: RV1106 в zero-copy отдает только int8.
    // _dims — размерности тензора, у выходов это [1, высота сетки, ширина сетки, каналы].
    // _n_dims — сколько размерностей заполнено в _dims.
    // _zp, _scale — параметры квантования: по ним int8 переводится обратно в float.
    struct tensor
    {
        const int8_t* _data;
        const uint32_t* _dims;
        uint32_t _n_dims;
        int32_t _zp;
        float _scale;
    };

    // Что br_npu узнал о модели при инициализации rknn.
    // _input_width, _input_height — размер входа модели (в него RGA вписывает кадр).
    // _input_channels — каналов во входе (3 для RGB).
    // _is_quant — выходы квантованы в int8: разбор работает только с такими.
    // _outputs, _output_count — выходные тензоры модели (головы).
    struct model_info
    {
        int _input_width;
        int _input_height;
        int _input_channels;
        bool _is_quant;
        const tensor* _outputs;
        int _output_count;
    };

    br_model() = default;
    virtual ~br_model() = default;

    // Настройки модели: общие для любой модели, поэтому живут в базовом классе
    void set_settings(const settings settings) { _settings = settings; }

    // Ниже то, что br_npu спрашивает у модели, чтобы создать контекст NPU и проверить блок кадра.
    // Путь до файла модели: br_npu отдает его в rknn_init
    const std::string& get_model_path() const { return _settings._path_to_model; }
    // Ширина входа модели: с ней сверяется вход загруженной модели
    uint16_t get_input_width() const { return _settings._width_model; }
    // Высота входа модели
    uint16_t get_input_height() const { return _settings._height_model; }

    // Проверить форму выходов модели и подготовить разбор.
    // model_info — то, что br_npu узнал о модели при инициализации rknn.
    // Возвращает false, если модель не подходит разбору: сообщение печатает реализация.
    virtual bool init(const model_info& model_info) = 0;

    // Разобрать выходы прогона в рамки в координатах кадра.
    // outputs — выходные тензоры прогона.
    // output_count — сколько их (у YOLOv5 это число голов).
    // detections — куда сложить результат, старый список очищается.
    virtual bool process(const tensor* outputs, const int output_count, detection_list& detections) = 0;

    // Название класса по его номеру.
    // cls_id — номер класса из модели, доставшийся от разбора.
    virtual const char* get_class_name(const int cls_id) const = 0;

protected:
    settings _settings;
};

// Реализация разбора под тип модели из конфига. nullptr, если тип неизвестен
std::unique_ptr<br_model> br_model_create(const std::string& type);
