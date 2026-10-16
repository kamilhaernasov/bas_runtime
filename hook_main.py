#!/usr/bin/env python3
"""
hook_main.py - обвязка внешнего скрипта (хука) bas_runtime.

Рантайм запускает этот файл, а пользовательский скрипт подхватывается здесь: скрипт видит только
кадр и боксы детекции, все остальное - разделяемая память и ожидание кадров - тут. От
пользовательского скрипта нужна одна функция:

    draw(frame, detections)

Аргументы подставляет рантайм:

    hook_main.py <дескриптор разделяемой памяти> <размер памяти> <пользовательский скрипт>

Разделяемая память - POSIX-объект в /dev/shm, у которого рантайм сразу снял имя: работаем
только по дескриптору, ничего монтировать и убирать за собой не нужно. Раскладка описана в br_hook_shared.h - правишь
там, правь и здесь.

Что внутри памяти (смещения ниже):

    width, height, frame_fd          - размер кадра и дескриптор буфера с ним
    script_pid                       - сюда пишем свой pid, когда поднялись: до этого
                                       рантайм считает, что мы еще запускаемся
    state                            - что происходит: free, filled или answered
    det_count, boxes[64]             - сколько боксов и сами боксы: координаты,
                                       уверенность, имя класса

Обмен: рантайм кладет кадр и боксы и последним ставит state = FILLED; мы видим FILLED, рисуем
прямо в буфере и последним ставим state = ANSWERED; рантайм забирает кадр и ставит FREE.
Пока state != FREE, рантайм кадр не отдает, и в поток идут текущие кадры с камеры - поэтому
отвечать надо быстро: все, что скрипт не успел, в этот кадр не попадет.

Кадр лежит в буфере без копий: mmap по дескриптору из памяти и numpy-вид (height, width, 3)
uint8 в порядке RGB - рисуй прямо в него. В этом же порядке задаются цвета: (255, 0, 0) - красный,
(0, 0, 255) - синий, так что в cv2 цвета для кадра-скрипта - это RGB, а не привычный BGR.

Рантайм следит за mtime пользовательского скрипта и поднимает процесс заново на каждую правку.
Если скрипт упадет, рантайм переживет это и запустит его снова, а кадры будут идти без разметки.
"""

import importlib.util
import mmap
import os
import struct
import sys
import time

import numpy as np

# Раскладка разделяемой памяти: те же поля и в том же порядке, что в br_hook_shared.h
BOXES_MAX = 64
BOX = struct.Struct("<iiiif24s")

OFF_WIDTH = 0
OFF_HEIGHT = 4
OFF_FRAME_FD = 8
OFF_SCRIPT_PID = 12
OFF_STATE = 16
OFF_DET_COUNT = 20
OFF_BOXES = 24  # шесть 4-байтовых полей шапки

# Состояние обмена: как в br_hook_shared.h
STATE_FILLED = 1
STATE_ANSWERED = 2

# Как часто проверяем, не отдал ли рантайм новый кадр
POLL_SEC = 0.0002

# Сколько ждем кадров, прежде чем уйти: рантайм мог упасть, а держать буфер кадра незачем
IDLE_EXIT_SEC = 10.0


def load_user_script(path):
    """Пользовательский скрипт берем по пути из конфига: имя файла любое"""
    spec = importlib.util.spec_from_file_location("br_hook_user", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not callable(getattr(module, "draw", None)):
        raise RuntimeError(f"{path}: нужна функция draw(frame, detections)")

    return module


def make_detections(shared, count):
    """Боксы из памяти в список словарей: то, что рисует пользовательский скрипт"""
    detections = []

    for i in range(count):
        left, top, right, bottom, prop, name = BOX.unpack_from(shared, OFF_BOXES + i * BOX.size)
        detections.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "prop": prop,
                "name": name.split(b"\0", 1)[0].decode("utf-8", "replace"),
            }
        )

    return detections


def main():
    if len(sys.argv) != 4:
        print("hook: ожидаются аргументы <дескриптор памяти> <размер памяти> <скрипт>", file=sys.stderr, flush=True)
        return 1

    fd_shared, size_shared = (int(value) for value in sys.argv[1:3])
    shared = mmap.mmap(fd_shared, size_shared, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    width, height, fd_frame = struct.unpack_from("<III", shared, OFF_WIDTH)

    # Кадр: тот же буфер, что отдает рантайм. Пишем прямо в него, копий нет
    frame_map = mmap.mmap(fd_frame, width * height * 3, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=frame_map)

    try:
        user = load_user_script(sys.argv[3])
    except Exception as error:  # ошибка в скрипте пользователя: говорим о ней и выходим, рантайм поднимет заново
        print(f"hook: {error}", file=sys.stderr, flush=True)
        return 1

    # Отмечаемся, что поднялись и все разобрали: до этого рантайм не ждет от нас кадров
    struct.pack_into("<i", shared, OFF_SCRIPT_PID, os.getpid())
    print(f"hook: pid {os.getpid()}, кадр {width}x{height} RGB888, скрипт {sys.argv[3]}", flush=True)

    idle_start = time.monotonic()

    while True:
        if struct.unpack_from("<I", shared, OFF_STATE)[0] != STATE_FILLED:
            if time.monotonic() - idle_start > IDLE_EXIT_SEC:
                print("hook: кадров нет, уходим", flush=True)
                return 0
            time.sleep(POLL_SEC)
            continue

        idle_start = time.monotonic()
        count = min(struct.unpack_from("<I", shared, OFF_DET_COUNT)[0], BOXES_MAX)
        user.draw(frame, make_detections(shared, count))

        # Ответ: рантайм ждет его, чтобы освободить буфер. Состояние пишем последним
        struct.pack_into("<I", shared, OFF_STATE, STATE_ANSWERED)


if __name__ == "__main__":
    sys.exit(main())
