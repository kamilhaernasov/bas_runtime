#include <iostream>
#include "INIReader.h"
#include "fsm.h"

bool set_fsm_settings(fsm& f, const std::string& config_name);

int main(int argc, char *argv[])
{
    std::cout << "Start..." << std::endl;

    fsm bas_fsm;
    set_fsm_settings(bas_fsm, "config.ini");
    bas_fsm.init();
}

bool set_fsm_settings(fsm& f, const std::string& config_name)
{
    return false;
}