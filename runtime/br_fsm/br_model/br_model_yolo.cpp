#include "br_model_yolo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// Ниже порт postprocess.cc из примера luckfox_pico_rtsp_yolov5: деквантование выходных
// тензоров, разбор боксов и NMS. Наружу из модуля ничего не торчит, все внутренности
// доступны только методам br_model_yolo.
namespace
{
    // Сколько значений выходного тензора приходится на один бокс помимо оценок классов:
    // четыре координаты и уверенность объекта. Это формат головы YOLOv5, а не свойство
    // модели: тензор отдает только общее число каналов, где в них кончаются координаты,
    // знает только сам алгоритм разбора (см. индексы hw_ptr ниже)
    constexpr int BOX_AND_CONF_SIZE = 5;

    // Ограничение значения в диапазоне [min, max].
    // Если val < min = min
    // Если val > max = max
    // Иначе val без дробной части
    int32_t clamp_value(float val, int32_t min, int32_t max) { return val > min ? (val < max ? val : max) : min; }

    // Квантование float32 в int8
    // scale - масштабирование, zp - смещение (zero point)
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        return static_cast<int8_t>(clamp_value(dst_val, -128, 127));
    }

    // Деквантование int8 в float32
    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale; }
}

// Файл меток: по названию класса в строке, число строк задает число классов модели
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

// Файл якорей: числа парами (ширина, высота), по паре на якорь.
// Идут подряд по головам: сначала все якоря старшей головы, потом следующей.
// Пример содержимого: 10.0 13.0 16.0 30.0 33.0 23.0 ... — это первые три якоря YOLOv5
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
    // Разбор работает с целочисленными выходами: RV1106 в zero-copy отдает только i8
    if (!model_info._is_quant)
    {
        printf("%s: model output is not quantized!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Пороги приходят из конфига модели: без них разбор не имеет смысла
    if (_settings._box_thresh <= 0 || _settings._nms_thresh <= 0)
    {
        printf("%s: thresholds are not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Кадр во вход модели укладывает RGA, а разбор возвращает рамки обратно по его геометрии
    if (_settings._scale <= 0)
    {
        printf("%s: letterbox scale is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Сколько рамок помещается в кадр, задается настройкой: при нуле детекций не будет
    if (_settings._max_detections <= 0)
    {
        printf("%s: max detections is not set!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Названия классов и якоря голов лежат в файлах рядом с моделью: обе вещи относятся
    // именно к ней, поэтому задаются данными, а не константами
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
    const int anchor_count = (int)_anchors.size() / 2;
    if (anchor_count % _head_count != 0)
    {
        printf("%s: anchors %d do not match heads %d!\n", __PRETTY_FUNCTION__, anchor_count, _head_count);
        return false;
    }
    _anchor_per_branch = anchor_count / _head_count;

    // Каналы выхода: на каждую ячейку сетки по prop_box_size значений на каждый якорь.
    // Тензоры уже собраны базой из атрибутов модели, здесь берется форма первой головы
    const int channels = (int)_tensors[0]._dims[_tensors[0]._n_dims - 1];
    if (channels % _anchor_per_branch != 0)
    {
        printf("%s: output channels %d do not match anchors per head %d!\n",
               __PRETTY_FUNCTION__, channels, _anchor_per_branch);
        return false;
    }
    _prop_box_size = channels / _anchor_per_branch;

    // Классов столько, сколько строк в файле меток: имена классов и выход модели
    // описывают одну модель, поэтому их числа должны сойтись
    _class_num = (int)_labels.size();
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
    detections.clear();

    // Кандидаты со всех голов лежат в одном списке: дубликаты между головами тоже возможны.
    // Описатели голов собраны в init: на каждом прогоне меняется только содержимое их памяти
    std::vector<box> boxes;
    for (int head = 0; head < (int)_tensors.size(); head++)
    {
        process_head(_tensors[head], head, boxes);
    }

    // Ничего не нашли
    if (boxes.empty())
    {
        return true;
    }

    suppress_overlaps(boxes);

    for (const box& candidate : boxes)
    {
        // Сколько рамок помещается в кадр, задает настройка
        if ((int)detections.size() >= _settings._max_detections)
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
        result.cls_id = candidate.cls_id;
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

// Разбор одной головы модели: раскладка NHWC, деквантование и декодирование боксов.
// head — номер головы: по нему берутся якоря этой головы.
// output — выходной тензор головы.
// boxes — куда складываются найденные кандидаты.
//
// Раскладка выходного тензора: на каждую ячейку сетки приходится anchor_per_branch якорей,
// на каждый якорь prop_box_size значений:
// [0..3] — смещение центра и размер рамки относительно якоря
// [4]    — уверенность объекта
// [5..]  — оценки классов
// Поэтому одна ячейка занимает align_c = prop_box_size * anchor_per_branch значений
void br_model_yolo::process_head(const tensor& output, const int head, std::vector<box>& boxes) const
{
    // Высота и ширина сетки этой головы: у каждой головы свой шаг (stride)
    const int grid_h = output._dims[2];
    const int grid_w = output._dims[1];
    // Шаг сетки: сколько пикселей входа приходится на одну ячейку
    const int stride = _model_height / grid_h;
    // Параметры квантования: ими int8 переводится обратно в float
    const int32_t zp = output._zp;
    const float scale = output._scale;
    const int8_t* input = output._data;
    const float threshold = _settings._box_thresh;
    // Порог в квантованном виде: сравнивать int8 быстрее, чем float
    const int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    const int align_c = _prop_box_size * _anchor_per_branch;
    // Якоря этой головы: (ширина, высота) на каждый якорь
    const float* ptr_anchor = &_anchors[head * _anchor_per_branch * 2];

    for (int h = 0; h < grid_h; h++)
    {
        for (int w = 0; w < grid_w; w++)
        {
            for (int a = 0; a < _anchor_per_branch; a++)
            {
                // Смещение до набора значений этого якоря в этой ячейке сетки
                const int hw_offset = h * grid_w * align_c + w * align_c + a * _prop_box_size;
                const int8_t* hw_ptr = input + hw_offset;
                const int8_t box_confidence = hw_ptr[4];

                if (box_confidence >= thres_i8)
                {
                    // Берем самый вероятный класс: остальные для этой рамки не нужны
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

                    const float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    const float class_prob_f32 = deqnt_affine_to_f32(max_class_probs, zp, scale);
                    const float limit_score = box_conf_f32 * class_prob_f32;

                    if (limit_score > threshold)
                    {
                        // Рамка выдается относительно якоря: сигмоиды дают смещение центра
                        // внутри ячейки и размер, якорь переводит размер в пиксели входа
                        float box_x = deqnt_affine_to_f32(hw_ptr[0], zp, scale) * 2.0 - 0.5;
                        float box_y = deqnt_affine_to_f32(hw_ptr[1], zp, scale) * 2.0 - 0.5;
                        float box_w = deqnt_affine_to_f32(hw_ptr[2], zp, scale) * 2.0;
                        float box_h = deqnt_affine_to_f32(hw_ptr[3], zp, scale) * 2.0;
                        box_w = box_w * box_w;
                        box_h = box_h * box_h;

                        // Центр ячейки сетки плюс смещение, размер — от якоря головы
                        box_x = (box_x + w) * (float)stride;
                        box_y = (box_y + h) * (float)stride;
                        box_w *= ptr_anchor[a * 2];
                        box_h *= ptr_anchor[a * 2 + 1];

                        // Из центра и размера получаем углы рамки
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

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

// Подавление дубликатов (NMS): на один объект модель часто выдает несколько похожих рамок.
// Сравниваем только рамки одного класса: разные классы могут законно перекрываться
// (например, человек и его рюкзак).
//
// Порядок работы:
// 1. Сортируем кандидатов по убыванию уверенности: самые уверенные рамки сохраняем первыми.
// 2. Идем по списку и оставляем рамку, только если она пересекается слабее порога
//    _nms_thresh с каждой уже оставленной рамкой своего класса.
//
// Пример: три рамки вокруг одного человека с уверенностями 0.9, 0.7 и 0.5 — останется
// только первая, потому что две другие пересекаются с ней сильнее порога.
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

// Коэффициент пересечения двух рамок (IoU) = площадь пересечения / площадь объединения.
// Площадь пересечения — это площадь той области, где прямоугольники накладываются.
// Площадь объединения — это площадь всей области, которую занимают оба прямоугольника вместе (без двойного счёта общей части).
// a, b — рамки одного класса, которые сравниваются.
// Возвращает 0, если рамки не пересекаются, и 1, если совпадают.
float br_model_yolo::overlap(const box& a, const box& b)
{
    const float w = fmax(0.f, fmin(a.right, b.right) - fmax(a.left, b.left) + 1.0);
    const float h = fmax(0.f, fmin(a.bottom, b.bottom) - fmax(a.top, b.top) + 1.0);
    const float intersection = w * h;
    const float union_area = (a.right - a.left + 1.0) * (a.bottom - a.top + 1.0) +
                             (b.right - b.left + 1.0) * (b.bottom - b.top + 1.0) - intersection;
    return union_area <= 0.f ? 0.f : (intersection / union_area);
}
