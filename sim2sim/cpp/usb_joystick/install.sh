make clean
make 
sudo rmmod usb_joystick
sudo cp usb_joystick.ko /lib/modules/$(uname -r)/kernel/drivers/input/joystick/
sudo depmod -a
sudo insmod /lib/modules/$(uname -r)/kernel/drivers/input/joystick/usb_joystick.ko
