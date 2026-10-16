#include "br_hook.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits.h>
#include <string>
#include <filesystem>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace
{
    /* 
    Шаг ожидания ответа скрипта в синхронном режиме 
    */
    constexpr uint32_t HOOK_POLL_US = 200;

    /* 
    Пауза между попытками поднять упавший на старте скрипт 
    */
    constexpr uint64_t HOOK_RESTART_PAUSE_US = 1000000;

    /* 
    Как часто смотрим, не изменился ли файл скрипта 
    */
    constexpr uint64_t HOOK_MTIME_CHECK_US = 1000000;
}

br_hook::br_hook()
{

}

br_hook::~br_hook()
{

}

void br_hook::set_settings(const settings settings)
{
    _settings = settings;
    _limit_us_wait = _settings._timeout_ms * 1000;
}

uint64_t br_hook::get_current_time_us() const
{
    static struct timespec time = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000000 + time.tv_nsec / 1000;
}

std::string br_hook::get_main_script_path() const
{
    std::error_code ec;
    auto exe = std::filesystem::read_symlink("/proc/self/exe", ec); // symlink на процесс рантайма
    if (ec) 
    {
        // Чет не вышло.
        printf("%s: can't find runtime path!\n", __PRETTY_FUNCTION__);
        return "./hook_main.py";
    };
    // Возвращаем путь.
    return (exe.parent_path() / "hook_main.py").string();
}

bool br_hook::is_enabled() const
{
    return _settings._is_enabled;
}

bool br_hook::is_async() const
{
    return _settings._is_async;
}

bool br_hook::is_free() const
{
    if (!_settings._is_enabled || _shared == nullptr)
    {
        return false;
    }

    return _own_state == own::none;
}

bool br_hook::init(const uint16_t width, const uint16_t height, const MB_BLK ptr_blk_frame)
{
    // Если мы не включили скрипт
    if (!_settings._is_enabled)
    {
        printf("%s: br_hook disabled\n", __PRETTY_FUNCTION__);
        return true;
    }

    // Запоминаем указатель на наш блок с кадром под скрипт.
    _ptr_blk_frame = ptr_blk_frame;

    // Создаем shm. Имя фиксированное, объект переиспользуем: повторный запуск рантайма (или старт
    // после падения) не должен упираться в уже созданную память. Содержимое все равно зануляем ниже
    _size_shared = sizeof(br_hook_shared);
    _fd_shared = shm_open("/bas_runtime_hook", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    // Не получилось.
    if (_fd_shared < 0)
    {
        printf("%s: shm_open fail! Desc: %d\n", __PRETTY_FUNCTION__, errno);
        return false;
    }
    // Приводим к нужном размеру.
    if (ftruncate(_fd_shared, static_cast<off_t>(sizeof(br_hook_shared))) != 0)
    {
        printf("%s: ftruncate fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Мапим к процессу рантайма.
    void* ptr_shared = mmap(nullptr, _size_shared, PROT_READ | PROT_WRITE, MAP_SHARED, _fd_shared, 0);
    // Не получилось.
    if (ptr_shared == MAP_FAILED)
    {
        printf("%s: mmap fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Приводим void* к нормальному ptr на структуру.
    _shared = static_cast<br_hook_shared*>(ptr_shared);
    // Зануляем шару: объект в /dev/shm переиспользуется, поэтому прошлое состояние надо снять.
    // По полям, а не через memset по всей структуре: атомики затирать как обычную память нельзя
    _shared->_script_pid.store(0, std::memory_order_relaxed);
    _shared->_state.store(BR_HOOK_FREE, std::memory_order_relaxed);
    _shared->_det_count = 0;
    memset(_shared->_boxes, 0, sizeof(_shared->_boxes));

    // Достаем дескриптор на блок памяти под кадр скрипта.
    const int32_t fd_frame = RK_MPI_MB_Handle2Fd(_ptr_blk_frame);
    // Не получилось.
    if (fd_frame < 0)
    {
        printf("%s: RK_MPI_MB_Handle2Fd fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // Заполняем шарную память.
    _shared->_width = width; // ширина кадра с VI
    _shared->_height = height; // высота кадра с VI
    _shared->_frame_fd = fd_frame; // дескриптор на блок памяти под кадр скрипта

    // Стартуем скрипт. Не поднялся - не беда: рантайм работает как обычно, кадры идут в поток
    // без разметки, а poll() будет пробовать поднять скрипт заново
    if (!spawn_script())
    {
        printf("%s: can't start base script for hook: working without it!\n", __PRETTY_FUNCTION__);
    }

    return true;
}

bool br_hook::spawn_script()
{
    // Дескриптор на шарную память наследуем.
    if (fcntl(_fd_shared, F_SETFD, 0) != 0)
    {
        printf("%s: fcntl shared fail!\n", __PRETTY_FUNCTION__);
        return false;
    }
    // На кадр тоже. 
    if (fcntl(_shared->_frame_fd, F_SETFD, 0) != 0)
    {
        printf("%s: fcntl frame fail!\n", __PRETTY_FUNCTION__);
        return false;
    }

    // Тупая фигня, надо подумать, как проще передавать дескрипторы.
    const std::string fd_text = std::to_string(_fd_shared);
    const std::string size_text = std::to_string(_size_shared);

    // Путь до базового скрипта.
    const std::string path_main = get_main_script_path();

    // Аргументы старта.
    char* argv[] =
    {
        const_cast<char*>(_settings._path_python.c_str()), // путь до питона
        const_cast<char*>(path_main.c_str()), // путь до базового скрипта
        const_cast<char*>(fd_text.c_str()), // дескриптор до шарной памяти
        const_cast<char*>(size_text.c_str()), // размер шарной памяти
        const_cast<char*>(_settings._path_script.c_str()), // путь до юзер скрипта
        nullptr,
    };

    // Пытаемся запуститься.
    _pid = -1;
    const int32_t ret = posix_spawnp(&_pid, _settings._path_python.c_str(), nullptr, nullptr, argv, environ);
    if (ret != 0)
    {
        printf("%s: posix_spawnp fail! Desc: %d\n", __PRETTY_FUNCTION__, ret);
        _pid = -1;
    }

    // Достаем время последнего изменения, запуска и последного обновления времени изменения.
    struct stat info;
    _mtime_script = stat(_settings._path_script.c_str(), &info) == 0 ? info.st_mtime : 0;
    const uint64_t now_us = get_current_time_us();
    _time_us_spawn = now_us;
    _time_us_mtime = now_us;

    return _pid > 0;
}

bool br_hook::restart_script(const char* reason)
{
    printf("%s: %s: restart base hook (pid %d)\n", __PRETTY_FUNCTION__, reason, _pid);

    // Если у нас был старый процесс.
    if (_pid > 0)
    {
        // Убиваем.
        kill(_pid, SIGKILL);
        int status = 0;
        // Ждем, когда помрет.
        waitpid(_pid, &status, 0);
        _pid = -1;
    }

    // Если пришлось рестартить, вряд ли что-то путное произошло в скрипте. Сбрасываем все, что связано с кадром, в shm..
    reset_shared();
    // Скрипт не готов.
    _is_script_ready = false;
    // Сохраняем в шару нулевой pid.
    _shared->_script_pid.store(0, std::memory_order_relaxed);
    // Пытаемся рестартовать.
    return spawn_script();
}

bool br_hook::is_script_changed()
{
    struct stat info;
    // Если достать не получилось
    if (stat(_settings._path_script.c_str(), &info) != 0)
    {
        // Считаем, что не поменялось.
        return false;
    }

    return info.st_mtime != _mtime_script;
}

void br_hook::reset_shared()
{
    // Сбрасываем кол-во детекций
    _shared->_det_count = 0;
    // Выставляем состояние, что у скрипта ничего нет.
    _own_state = own::none;
    // Уведомляем скрипт, что он теперь свободен.
    _shared->_state.store(BR_HOOK_FREE, std::memory_order_release);
}

void br_hook::submit(const br_model::detection_list& detections)
{
    // Эм, проверяем, что мы не вышли за рамки диапазона принимаемых детекций. Шара не резиновая.
    const uint32_t count = detections.size() < BR_HOOK_BOXES_MAX ? detections.size() : BR_HOOK_BOXES_MAX;
    // Собираем детекции, если есть. 
    for (uint32_t i = 0; i < count; i++)
    {
        const br_model::detection& src = detections[i];
        br_hook_box& dst = _shared->_boxes[i];

        dst._left = src.box.left;
        dst._top = src.box.top;
        dst._right = src.box.right;
        dst._bottom = src.box.bottom;
        dst._prop = src.prop;
        strncpy(dst._name, src.cls_id.c_str(), BR_HOOK_CLASS_NAME_LEN - 1);
        dst._name[BR_HOOK_CLASS_NAME_LEN - 1] = 0;
    }
    // Уведомляем шару, сколько у нас детекций.
    _shared->_det_count = count;
    // Меняем состояние, что ждем ответ от скрипта.
    _own_state = own::script;
    // Запоминаем, когда положили кадр.
    _time_us_filled = get_current_time_us();

    // Всегда пишем последним. После этого начнет скрипт увидит кадр и детекции.
    _shared->_state.store(BR_HOOK_FILLED, std::memory_order_release);
}

bool br_hook::take_ready()
{
    if (!_settings._is_async || _own_state != own::script)
    {
        return false;
    }
    // Если скрипт не ответил, мы не готовы.
    if (_shared->_state.load(std::memory_order_acquire) != BR_HOOK_ANSWERED)
    {
        return false;
    }
    // Сбрасываем все, что связано с кадром, в shm.
    reset_shared();
    // Мы готовы двигаться к новому кадру.
    return true;
}

bool br_hook::wait()
{
    // Время начала запомнили.
    const uint64_t start_us = get_current_time_us();
    // Пуллим статус до таймаута.
    while (get_current_time_us() - start_us < _limit_us_wait)
    {
        if (_shared->_state.load(std::memory_order_acquire) == BR_HOOK_ANSWERED)
        {
            reset_shared();

            return true;
        }
        // Чтобы не жечь CPU зазря.
        usleep(HOOK_POLL_US);
    }

    // Таймаут, откидываем кадр.
    _own_state = own::dropped;

    return false;
}

void br_hook::poll()
{
    if (!_settings._is_enabled || _shared == nullptr)
    {
        return;
    }
    // Текущее время.
    const uint64_t now_us = get_current_time_us();

    // Если скрипт был
    if (_pid > 0)
    {
        int status = 0;
        // Если скрипт бац и завершился
        if (waitpid(_pid, &status, WNOHANG) == _pid)
        {
            printf("%s: base hook (pid %d) ended!\n", __PRETTY_FUNCTION__, _pid);
            _pid = -1;
        }
    }
    // Скрипта и не было...
    if (_pid < 0)
    {
        // Пробуем рестарт.
        if (now_us - _time_us_spawn >= HOOK_RESTART_PAUSE_US)
        {
            restart_script("base hook don't started!");
        }
        return;
    }

    // Ждем, пока скрипт поднимется за отведенное ему время
    if (_shared->_script_pid.load(std::memory_order_acquire) != _pid)
    {
        if (now_us - _time_us_spawn > (uint64_t)_settings._start_ms * 1000)
        {
            restart_script("base hook can't run!");
        }
        return;
    }
    // Если скрипт был не готов
    if (!_is_script_ready)
    {
        _is_script_ready = true;
        // Если он что-то делал
        if (_own_state != own::none)
        {
            reset_shared();
        }
        printf("%s: base hook (pid %d) alive...\n", __PRETTY_FUNCTION__, _pid);
    }

    // Надо дропнуть кадр.
    if (_own_state == own::dropped &&
        _shared->_state.load(std::memory_order_acquire) == BR_HOOK_ANSWERED)
    {
        reset_shared();
    }

    // Скрипт завис.
    if (_own_state != own::none && now_us - _time_us_filled > (uint64_t)_settings._stuck_ms * 1000)
    {
        restart_script("base hook stucked!");
        return;
    }

    // Проверка, что файл-скрипт юзера поменялся
    if (now_us - _time_us_mtime >= HOOK_MTIME_CHECK_US)
    {
        _time_us_mtime = now_us;
        if (is_script_changed())
        {
            restart_script("user-script changed!");
        }
    }
}

bool br_hook::release()
{
    if (!_settings._is_enabled)
    {
        return true;
    }

    if (_pid > 0)
    {
        kill(_pid, SIGKILL);
        int status = 0;
        waitpid(_pid, &status, 0);
        _pid = -1;
    }

    if (_shared != nullptr)
    {
        munmap(_shared, _size_shared);
        _shared = nullptr;
    }
    if (_fd_shared >= 0)
    {
        close(_fd_shared);
        _fd_shared = -1;
    }

    return true;
}
