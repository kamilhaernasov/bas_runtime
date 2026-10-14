#include "br_model_yolo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

/* 
Ниже порт postprocess.cc из примера luckfox_pico_rtsp_yolov5.
*/
namespace
{
    /*
    База. Сколько значений без уверенностей на каждый класс детекции приходится на каждый якорь.
    */
    constexpr int BOX_AND_CONF_SIZE = 5;

    /* 
    Ограничение значения в диапазоне [min, max].
    Если val < min = min
    Если val > max = max
    Иначе val без дробной части
    */
    int32_t clamp_value(float val, int32_t min, int32_t max) { return val > min ? (val < max ? val : max) : min; }

    /*
    Квантование float32 в int8
    scale - масштабирование, zp - смещение (zero point)
    */
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        return static_cast<int8_t>(clamp_value(dst_val, -128, 127));
    }

    /* 
    Деквантование int8 в float32
    */
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale; }
}

bool br_model_yolo::load_labels(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        printf("%s: open %s fail!\n", __PRETTY_FUNCTION__, path.c_str());
        return false;
    }

    _labels.clear();
    std::string line;
    while (std::getline(file, line))
    {
        // В файле могут быть виндовые переводы строк
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            _labels.push_back(line);
        }
    }

    printf("%s: %s: %u classes\n", __PRETTY_FUNCTION__, path.c_str(), (unsigned)_labels.size());

    return !_labels.empty();
}

bool br_model_yolo::load_anchors(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        printf("%s: open %s fail!\n", __PRETTY_FUNCTION__, path.c_str());
        return false;
    }

    _anchors.clear();
    std::string line;
    while (std::getline(file, line))
    {
        const char* cursor = line.c_str();
        while (*cursor != '\0')
        {
            char* end = nullptr;
            const double value = strtod(cursor, &end);
            if (end == cursor)
            {
                break;
            }
            _anchors.push_back(static_cast<float>(value));
            cursor = end;
        }
    }

    printf("%s: %s: %u values\n", __PRETTY_FUNCTION__, path.c_str(), (unsigned)_anchors.size());

    return !_anchors.empty();
}

bool br_model_yolo::setup(const model_info& model_info)
{
    // Проверям, что модель квантованная
    if (!model_info._is_quant)
    {
        printf("%s: model output is not quantized!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Проверяем, что стоят пороги на детекцию и перекрытие
    if (_settings._box_thresh <= 0 || _settings._nms_thresh <= 0)
    {
        printf("%s: thresholds are not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Проверяем, что кадр вписан 
    if (_settings._scale <= 0)
    {
        printf("%s: letterbox scale is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Проверяем, что нам разрешено вообще детектировать больше нуля объектов
    if (_settings._max_detections <= 0)
    {
        printf("%s: max detections is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Грузим классы и якоря
    if (!load_labels(_settings._path_to_labels) || !load_anchors(_settings._path_to_anchors))
    {
        return false;
    }

    // Число голов — это число выходных тензоров модели
    _head_count = model_info._output_count;
    if (_head_count <= 0)
    {
        printf("%s: model has no outputs!\n", __PRETTY_FUNCTION__);
        return false;
    }
    _model_height = model_info._input_height;

    // Якоря лежат парами (ширина, высота) и поделены между головами поровну
    const size_t anchor_count = _anchors.size() / 2;
    if (anchor_count % _head_count != 0)
    {
        printf("%s: anchors %d do not match heads %d!\n", __PRETTY_FUNCTION__, anchor_count, _head_count);
        return false;
    }
    // Количество якорей на одну ячейку в голове.
    _anchor_per_branch = anchor_count / _head_count;

    // Рассчитываем, сколько каналов значений есть в выходной ячейке.
    // Каналы - описание бокса детекции на всю ячейку.
    // В целом мы уже знаем это из модели, здесь же просто достаем из готового первого выходного тензора.
    const int channels = (int)_tensors[0]._dims[_tensors[0]._n_dims - 1];
    if (channels % _anchor_per_branch != 0)
    {
        printf("%s: output channels %d do not match anchors per head %d!\n",
               __PRETTY_FUNCTION__, channels, _anchor_per_branch);
        return false;
    }
    // Рассчитываем, сколько значений используются для описания одного бокса детекции.
    // Channels - это на всю ячейку. Внутри ячейки есть якоря. Собственно, поэтому и деление.
    _prop_box_size = channels / _anchor_per_branch;

    // На всякий пожарный проверяем, что вычисленное значение кол-во байт на один якорь сходится с ожидаемым:
    // 4 координаты + 1 уверенность, что объект есть + N уверенностей на каждый класс детекции
    _class_num = _labels.size();
    if (_prop_box_size != _class_num + BOX_AND_CONF_SIZE)
    {
        printf("%s: classes mismatch: model %d, %s %d!\n",
               __PRETTY_FUNCTION__, _prop_box_size - BOX_AND_CONF_SIZE,
               _settings._path_to_labels.c_str(), _class_num);
        return false;
    }

    printf("%s: heads=%d, anchors per head=%d, box size=%d, classes=%d\n",
           __PRETTY_FUNCTION__, _head_count, _anchor_per_branch, _prop_box_size, _class_num);

    return true;
}

bool br_model_yolo::process(detection_list& detections)
{
    // Во-первых, само собой чистим, что нам передали. Мало ли.
    detections.clear();

    // Боксы детекции всех якорей со всех ячеек, со всех голов.
    std::vector<box> boxes;
    // Проходимся по всем головам
    for (size_t head = 0; head < _tensors.size(); head++)
    {
        // Достаем все с головы
        process_head(_tensors[head], head, boxes);
    }

    // Ничего не нашли
    if (boxes.empty())
    {
        return true;
    }

    // Подавляем наложения
    suppress_overlaps(boxes);

    // Собираем итог детекций
    for (const box& candidate : boxes)
    {
        // Ограничиваемся настройками
        if (detections.size() >= _settings._max_detections)
        {
            break;
        }

        // Из пространства входа модели в координаты кадра: убираем поля letterbox
        // и делим на масштаб, которым кадр вписывали во вход
        detection result;
        result.box.left = clamp_value((candidate.left - _settings._pad_x) / _settings._scale, 0, _settings._width - 1);
        result.box.top = clamp_value((candidate.top - _settings._pad_y) / _settings._scale, 0, _settings._height - 1);
        result.box.right = clamp_value((candidate.right - _settings._pad_x) / _settings._scale, 0, _settings._width - 1);
        result.box.bottom = clamp_value((candidate.bottom - _settings._pad_y) / _settings._scale, 0, _settings._height - 1);
        result.prop = candidate.score;
        result.cls_id = get_class_name(candidate.cls_id);

        detections.push_back(result);
    }

    return true;
}

const char* br_model_yolo::get_class_name(const int cls_id) const
{
    static char null_str[] = "null";

    if (cls_id < 0 || cls_id >= (int)_labels.size())
    {
        return null_str;
    }

    return _labels[cls_id].c_str();
}

void br_model_yolo::process_head(const tensor& output, const size_t head, std::vector<box>& boxes) const
{
    // TODO: добавить проверки на границы.

    // Дэм, достаем размеры сетки у этого тензора. NHWC, помним.
    const int grid_h = output._dims[1]; // nHwc
    const int grid_w = output._dims[2]; // nhWc
    // Шаг сетки: сколько пикселей входа приходится на одну ячейку
    const int stride = _model_height / grid_h;
    // Достаем параметры квантования
    const int32_t zp = output._zp;
    const float scale = output._scale;
    // Достаем сами данные
    const int8_t* input = output._data;
    // Порог в квантованном виде: сравнивать int8 быстрее, чем float
    const int8_t thres_i8 = qnt_f32_to_affine(_settings._box_thresh, zp, scale);
    // Сколько в ячейке параметров у нас для описания каждого бокса в каждом якоре
    const int align_c = _prop_box_size * _anchor_per_branch;
    // Якоря этой головы: (ширина, высота) на каждый якорь (умножаем на 2, так одна ячейка массива описывает весь якорь)
    const float* ptr_anchor = &_anchors[head * _anchor_per_branch * 2];

    // Ходим по строкам сетки
    for (int h = 0; h < grid_h; h++)
    {
        // Ходим по столбцам сетки
        for (int w = 0; w < grid_w; w++)
        {
            // Ходим по якорям
            for (int a = 0; a < _anchor_per_branch; a++)
            {
                // Смещение до самих значений детекции внутри якоря
                // строка_сетки*ширину_сетки*кол-во_значений_одной_ячейки + столбец_сетки * кол-во_значений_одной_ячейки + якорь * кол-во_значений_в_одном_якоре
                const int hw_offset = h * grid_w * align_c + w * align_c + a * _prop_box_size;
                // Сдвигаем указатель с начала на смещение до данных
                const int8_t* hw_ptr = input + hw_offset;
                // Уверенность, что объект есть (отдается в квантованном виде)
                const int8_t box_confidence = hw_ptr[4];
                // Если чёт есть в этом якоре
                if (box_confidence >= thres_i8)
                {
                    // Ходим по классам и берем класс с максимальной уверенностью
                    int8_t max_class_probs = hw_ptr[BOX_AND_CONF_SIZE];
                    int max_class_id = 0;
                    for (int k = 1; k < _class_num; ++k)
                    {
                        const int8_t prob = hw_ptr[BOX_AND_CONF_SIZE + k];
                        if (prob > max_class_probs)
                        {
                            max_class_id = k;
                            max_class_probs = prob;
                        }
                    }
                    // Деквантируем уверенность, что что-то есть
                    const float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    // Деквантируем уверенность конкретного класса-победителя
                    const float class_prob_f32 = deqnt_affine_to_f32(max_class_probs, zp, scale);
                    // Вычисляем вероятность детекции
                    const float limit_score = box_conf_f32 * class_prob_f32;
                    // Если вероятность больше порога
                    if (limit_score > _settings._box_thresh)
                    {
                        // Не пытался это понять, взял из примера. Что-то про сырой вывод yolo.
                        float box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0 - 0.5;
                        float box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0 - 0.5;
                        float box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0;
                        float box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;

                        // Центр ячейки сетки плюс смещение, размер — от якоря головы
                        box_x = (box_x + w) * stride;
                        box_y = (box_y + h) * stride;
                        box_w *= ptr_anchor[a * 2];
                        box_h *= ptr_anchor[a * 2 + 1];

                        // Из центра и размера получаем углы рамки
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        // Ура, у нас есть кандидат.
                        box candidate;
                        candidate.left = box_x;
                        candidate.top = box_y;
                        candidate.right = box_x + box_w;
                        candidate.bottom = box_y + box_h;
                        candidate.score = limit_score;
                        candidate.cls_id = max_class_id;
                        boxes.push_back(candidate);
                    }
                }
            }
        }
    }
}

void br_model_yolo::suppress_overlaps(std::vector<box>& boxes) const
{
    std::sort(boxes.begin(), boxes.end(),
              [](const box& a, const box& b) { return a.score > b.score; });

    std::vector<box> kept;
    kept.reserve(boxes.size());

    for (const box& candidate : boxes)
    {
        bool is_overlapped = false;
        for (const box& kept_box : kept)
        {
            if (kept_box.cls_id == candidate.cls_id && overlap(kept_box, candidate) > _settings._nms_thresh)
            {
                is_overlapped = true;
                break;
            }
        }

        if (!is_overlapped)
        {
            kept.push_back(candidate);
        }
    }

    boxes.swap(kept);
}

float br_model_yolo::overlap(const box& a, const box& b)
{
    const float w = fmax(0.f, fmin(a.right, b.right) - fmax(a.left, b.left) + 1.0);
    const float h = fmax(0.f, fmin(a.bottom, b.bottom) - fmax(a.top, b.top) + 1.0);
    const float intersection = w * h;
    const float union_area = (a.right - a.left + 1.0) * (a.bottom - a.top + 1.0) +
                             (b.right - b.left + 1.0) * (b.bottom - b.top + 1.0) - intersection;
    return union_area <= 0.f ? 0.f : (intersection / union_area);
}
