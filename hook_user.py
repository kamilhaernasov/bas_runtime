#!/usr/bin/env python3
"""
hook_user.py - пример пользовательского скрипта bas_runtime: рисует векторы движения по кадрам
(оптический поток). Детекции этому скрипту не нужны вовсе: векторы считаются по самой картинке,
поэтому обновляются на каждом кадре.

Рантайм отдает кадр на каждом кадре потока. Скрипт сравнивает текущий кадр с предыдущим:
на сетке точек считает, куда сместилась картинка (cv2.calcOpticalFlowPyrLK), и рисует стрелки
на части точек, где смещение заметно.

От скрипта обвязке нужна одна функция:

    draw(frame, detections)

    frame       - кадр камеры numpy (height, width, 3) uint8, порядок каналов RGB: рисуй прямо
                  в него. Это вид в буфер рантайма, и буфер будет перезаписан следующим кадром,
                  поэтому сравнивать кадры можно только со своей копией - здесь для этого
                  каждый кадр считается уменьшенный серый кадр (своя память);
    detections  - список найденных объектов: left, top, right, bottom, prop, name. Здесь не
                  используется и может быть пустым.

Ручки: SCALE - во сколько раз уменьшать кадр для расчета потока, GRID_STEP - шаг сетки точек,
DRAW_STEP - какую часть стрелок рисовать, MIN_MOVE_PX - порог шума.

Бюджет: кадров 30 в секунду, значит скрипту желательно укладываться в ~30 мс. Поэтому поток
считается на уменьшенном кадре (SCALE) и по сетке точек (GRID_STEP): на полном 1080p оптический
поток съел бы весь бюджет. Ориентир: при SCALE = 4 и GRID_STEP = 32 на 1080p получается
480x270 и ~120 точек - это единицы миллисекунд на быстрой машине; на этой плате крутите ручки
в сторону SCALE = 8 или большего GRID_STEP, если не укладываетесь.

Что рисовать - дело скрипта: рантайм и обвязка в отрисовку не лезут. Fps здесь считается
и рисуется самим скриптом - как пример того, что можно вывести поверх кадра.

Файл можно править на ходу: рантайм поднимет процесс заново на каждую правку.
"""

import math
import time

import cv2
import numpy as np

# --- ручки потока и отрисовки ---
# Цвета задаются в порядке кадра - RGB. Мнемоника та же, что у cv2, только R и B местами
COLOR_VECTOR = (255, 200, 0)  # стрелка вектора движения
COLOR_FPS = (255, 255, 255)
SCALE = 4  # во сколько раз уменьшаем кадр для расчета: 4 - это 480x270 с 1920x1080
GRID_STEP = 32  # шаг сетки точек на уменьшенном кадре: больше шаг - дешевле кадр
MIN_MOVE_PX = 3.0  # смещение меньше этого (в пикселях полного кадра) не рисуем: шум
WIN_SIZE = 21  # окно поиска LK на уменьшенном кадре
DRAW_STEP = 4  # рисовать каждую N-ю точку сетки: меньше стрелок на экране, а картина та же

fps = 0.0
last_time = 0.0

_previous_gray = None
_grid = None


def _make_grid(width, height):
    """Сетка точек под векторы: по одной в каждой клетке GRID_STEP уменьшенного кадра"""
    xs = np.arange(GRID_STEP // 2, width, GRID_STEP, dtype=np.float32)
    ys = np.arange(GRID_STEP // 2, height, GRID_STEP, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)

    return grid.reshape(-1, 1, 2)


def _draw_vectors(frame, previous_gray, gray):
    """Считает смещение картинки на сетке точек и рисует по стрелке на каждую точку"""
    global _grid

    if _grid is None or _previous_gray.shape != gray.shape:
        _grid = _make_grid(gray.shape[1], gray.shape[0])

    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, _grid, None,
                                                winSize=(WIN_SIZE, WIN_SIZE), maxLevel=2)
    if moved is None:
        return

    # Векторы рисуем не на каждой точке сетки, а через DRAW_STEP: на 1080p стрелок и так много,
    # а по картинке видно то же самое
    for i, (from_point, to_point) in enumerate(zip(_grid, moved)):
        if status[i][0] == 0 or (i % DRAW_STEP) != 0:
            continue

        # Обратно в пиксели полного кадра
        dx = (to_point[0][0] - from_point[0][0]) * SCALE
        dy = (to_point[0][1] - from_point[0][1]) * SCALE
        if math.hypot(dx, dy) < MIN_MOVE_PX:
            continue

        cv2.arrowedLine(frame, (int(from_point[0][0] * SCALE), int(from_point[0][1] * SCALE)),
                        (int(to_point[0][0] * SCALE), int(to_point[0][1] * SCALE)),
                        COLOR_VECTOR, 2, cv2.LINE_AA, tipLength=0.4)


def draw(frame, detections):
    global fps, last_time, _previous_gray

    now = time.monotonic()
    if last_time:
        instant = 1.0 / max(now - last_time, 1e-6)
        fps = instant if fps == 0.0 else fps * 0.9 + instant * 0.1
    last_time = now

    # Свой маленький серый кадр: он наш, поэтому его можно держать до следующего кадра.
    # Интерполяция INTER_LINEAR, а не INTER_AREA: площадь по кадру 1080p на этом процессоре
    # стоит дороже всего остального в скрипте
    small = cv2.resize(frame, None, fx=1.0 / SCALE, fy=1.0 / SCALE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if _previous_gray is not None and _previous_gray.shape == gray.shape:
        _draw_vectors(frame, _previous_gray, gray)
    _previous_gray = gray

    if fps:
        cv2.putText(frame, f"{fps:.1f} FPS", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FPS, 2)
