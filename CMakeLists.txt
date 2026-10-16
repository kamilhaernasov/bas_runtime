# Главный CMake под проект runtime для BAS.
#
# Структура, необходимая перед стартом:
# ├── build.sh -> project/build.sh ---- SDK compilation script
# ├── media --------------------------- Multimedia codec, ISP, and algorithm-related files (can be compiled as an independent SDK)
# ├── sysdrv -------------------------- U-Boot, kernel, and rootfs directories (can be compiled as an independent SDK)
# ├── project ------------------------- Reference applications, compilation configuration, and script directories
# ├── output -------------------------- Directory for generated image files after SDK compilation
# ├── tools --------------------------- Image packaging tools and flashing tools
# └── dependencies -------------------- Зависимости из example
#     ├── 3rdparty 
#     ├── include 
#     ├── lib
# Env: 
#    LUCKFOX_SDK_PATH - путь до SDK
# SDK: git clone https://github.com/LuckfoxTECH/luckfox-pico.git
# Example: https://github.com/LuckfoxTECH/luckfox_pico_rkmpi_example/tree/kernel-5.10.160
# Warning! Если ветки обновятся, добейтесь синхронизации между собираемым ядром из SDK и примерами. 
#
# Запуск сборки (из директории с CMakeLists.txt):
# cmake -S . -B build && cmake --build build && cmake --install build
# Дебаг:
# cmake -S . -B build -DDEBUG_FRAME=ON && cmake --build build && cmake --install build


cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME bas_runtime)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "Install prefix")

# ---- SDK & PATH ----
# Компиляторы задаются до project(): языки включаются вместе с компиляторами, позже
# выставленный CMAKE_C_COMPILER/CMAKE_CXX_COMPILER уже не влияет на сборку.
# Path to SDK from env
set(SDK_PATH $ENV{LUCKFOX_SDK_PATH})
# Без пути до SDK не найти ни компиляторы, ни заголовки Rockchip. Если сборка уже
# конфигурировалась раньше, ошибка вылезет не здесь, а где-то в середине сборки
# непонятным "rk_comm_mb.h: No such file or directory", поэтому проверяем сразу
if (NOT SDK_PATH)
    message(FATAL_ERROR
        "LUCKFOX_SDK_PATH не задан.\n"
        "Экспортируйте путь до SDK перед сборкой, например:\n"
        "    export LUCKFOX_SDK_PATH=$HOME/luckfox-pico")
endif()
# Path to compilers from SDK
set(CMAKE_C_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

project(bas_runtime
    # Версия проекта доезжает до кода как BAS_RUNTIME_VERSION
    VERSION 0.1
    LANGUAGES C CXX
)

# ---- C++ settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CMake settings ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Exe-xe-xe ----
add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/runtime/main.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_fsm.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_mb_pool/br_mb_pool.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_vi/br_vi.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_venc/br_venc.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_rtsp/br_rtsp.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_rga/br_rga.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_npu/br_npu.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_model/br_model.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_model/br_model_yolo.cpp
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_hook/br_hook.cpp
)

# ---- Libs ----
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    RV1106_1103
    BAS_RUNTIME_VERSION="${PROJECT_VERSION}"
)

# ---- Debug ----
# Отладка кадров: под определением debug_frame печатается строка на каждый кадр.
# Ручка выключена по умолчанию, включается ключом -DDEBUG_FRAME=ON
option(DEBUG_FRAME "Print per-frame debug: stage times and detection count" OFF)
if (DEBUG_FRAME)
    target_compile_definitions(${PROJECT_NAME} PRIVATE debug_frame)
endif()

target_link_directories(${PROJECT_NAME} PRIVATE
    "${SDK_PATH}/dependencies/lib/uclibc"
)
# Pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
# Inih
add_library(inih STATIC
    ${CMAKE_SOURCE_DIR}/externals/inih/ini.c
    ${CMAKE_SOURCE_DIR}/externals/inih/cpp/INIReader.cpp
)
target_include_directories(inih PUBLIC
    ${CMAKE_SOURCE_DIR}/externals/inih
    ${CMAKE_SOURCE_DIR}/externals/inih/cpp
)
# Строка длиннее INI_MAX_LINE (по умолчанию 200 байт) для inih — ошибка разбора.
# В config.ini комментарии на русском, а это 2 байта на символ, поэтому лимит поднят
target_compile_definitions(inih PRIVATE INI_MAX_LINE=1024)

# ---- Options ----
# Compiler
# -O2 обязателен: без него разбор выходов NPU (декод боксов и NMS) считается без оптимизации
target_compile_options(${PROJECT_NAME} PRIVATE
    -O2 -g -Wall -DISP_HW_V30 -DRKPLATFORM=ON -DARCH64=OFF
    -DROCKIVA -DUAPI2
    -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
)
# Linker
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${SDK_PATH}/dependencies/lib")

# ---- Link libs ----
target_link_libraries(${PROJECT_NAME}
    rknnmrt
    Threads::Threads
    rockiva
    sample_comm
    rockit
    rockchip_mpp
    rkaiq
    pthread
    rtsp
    rga
    inih
)

# std::filesystem в libstdc++ до GCC 9 лежит отдельной библиотекой: без нее не слинкуется
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

# ---- Include dirs ----
target_include_directories(${PROJECT_NAME} PRIVATE       
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_mb_pool
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_vi  
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_venc
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_rtsp     
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_rga     
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_npu                          
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_model
    ${CMAKE_SOURCE_DIR}/runtime/br_fsm/br_hook
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/common 
    ${CMAKE_CURRENT_SOURCE_DIR}/common/isp3.x 
    ${SDK_PATH}/media/iva/out/include
    ${SDK_PATH}/dependencies/include
    ${SDK_PATH}/dependencies/include/rknn
    ${SDK_PATH}/dependencies/include/librga 
    ${SDK_PATH}/dependencies/include/rkaiq
    ${SDK_PATH}/dependencies/include/rkaiq/uAPI2
    ${SDK_PATH}/dependencies/include/rkaiq/common
    ${SDK_PATH}/dependencies/include/rkaiq/xcore
    ${SDK_PATH}/dependencies/include/rkaiq/algos
    ${SDK_PATH}/dependencies/include/rkaiq/iq_parser
    ${SDK_PATH}/dependencies/include/rkaiq/iq_parser_v2
    ${SDK_PATH}/dependencies/include/rkaiq/smartIr
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
# Конфиги читаются из рабочего каталога, поэтому кладем их рядом с бинарником
install(FILES ${CMAKE_SOURCE_DIR}/config.ini DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/model.ini DESTINATION .)
# Конфиг и скрипты хука: лежат рядом с бинарником и запускаются по своим путям
install(FILES ${CMAKE_SOURCE_DIR}/hook.ini DESTINATION .)
# Обвязка хука: рантайм находит ее рядом с собой, а она - пользовательский скрипт из конфига
install(FILES ${CMAKE_SOURCE_DIR}/hook_main.py DESTINATION .)
# Пример пользовательского скрипта: одна функция draw(frame, detections)
install(FILES ${CMAKE_SOURCE_DIR}/hook_user.py DESTINATION .)