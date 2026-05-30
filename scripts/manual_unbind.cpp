#include "nvm_ctrl.h"
#include "ctrl.h"
#include "ioctl.h"

#include <fcntl.h>
#include <iostream>
#include <vector>



int main(){
    std::string path = "/dev/snvm_control"; 
    std::vector<struct pci_device_addr> to_delete = {
        {0, 0x4c, 0,0},
        {0, 0x50, 0,0}
    };

    int fd = open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0){
        std::cerr << "Failed to open control descriptor" << std::endl;
        return -1;
    }

    for (auto &addr : to_delete){
        int status = ioctl(fd, SNVM_DEVICE_UNBIND, &addr);
        if (status != 0){
            std::cerr << "Failed to unbind device" << std::endl;

        }
        status = nvm_chrdev_remove(fd, &addr);
        if (status != 0){
            std::cerr << "Failed to remove device descriptor" << std::endl;
        }
    }
    
    
    return 0;
}