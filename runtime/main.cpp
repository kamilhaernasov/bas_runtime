#include <iostream>
//#include "INIReader.h"
#include "br_fsm.h"

bool set_fsm_settings(br_fsm& f, const std::string& config_name);

int main(int argc, char *argv[])
{
    std::cout << "Start..." << std::endl;

    br_fsm bas_fsm;
    set_fsm_settings(bas_fsm, "config.ini");
    
    if (bas_fsm.init())
    {
        bas_fsm.start();
    }

    bas_fsm.release();
}

bool set_fsm_settings(br_fsm& f, const std::string& config_name)
{
    return false;
}