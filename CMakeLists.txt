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

cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME bas_runtime)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "Install prefix")
project(bas_runtime
    LANGUAGES CXX
)

# ---- SDK & PATH ----
# Path to SDK from env
set(SDK_PATH $ENV{LUCKFOX_SDK_PATH})
# Path to compilers from SDK
set(CMAKE_C_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

# ---- C++ settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CMake settings ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Exe-xe-xe ----
add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/runtime/src/main.cpp)

# ---- Libs ----
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    DRV1106_1103
)
target_link_directories(${PROJECT_NAME} PRIVATE
    "${SDK_PATH}/dependencies/lib/uclibc"
)
# OpenCV
set(OpenCV_DIR "${SDK_PATH}/dependencies/lib/uclibc/lib/cmake/opencv4")
find_package(OpenCV REQUIRED QUIET)
# Pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---- Options ----
# Compiler
target_compile_options(${PROJECT_NAME} PRIVATE
    -g -Wall -DISP_HW_V30 -DRKPLATFORM=ON -DARCH64=OFF
    -DROCKIVA -DUAPI2
    -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
)
# Linker
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${SDK_PATH}/dependencies/lib")

# ---- Link libs ----
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
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
)

# ---- Include dirs ----
target_include_directories(${PROJECT_NAME} PRIVATE                        
    ${OpenCV_INCLUDE_DIRS}  
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