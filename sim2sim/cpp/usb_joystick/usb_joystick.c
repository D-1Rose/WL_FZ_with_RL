#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/usb.h>
#include <linux/input.h>
#include <linux/hid.h>

#define USB_VENDOR_ID 0x413d  // 替换为你的USB设备的Vendor ID
#define USB_PRODUCT_ID 0x2104 // 替换为你的USB设备的Product ID
#define BUFFER_SIZE 64        // 增加缓冲区大小

struct usb_joystick
{
    struct usb_device *udev;
    struct input_dev *input;
    struct urb *irq;
    unsigned char *data;
    dma_addr_t data_dma;
};

static void usb_joystick_irq(struct urb *urb)
{
    struct usb_joystick *js = urb->context;
    int retval;

    if (urb->status)
    {
        printk(KERN_ERR "usb_joystick_irq: URB status error: %d\n", urb->status);
        return;
    }

    // 数据处理并报告为输入事件
    int xx = js->data[2] & 1 ? 32767 : (js->data[2] & 2 ? -32768 : 0);
    int yy = js->data[2] & 4 ? -32767 : (js->data[2] & 8 ? 32768 : 0);

    int lt = (js->data[4] - 0x80) * 256;
    int rt = (js->data[5] - 0x80) * 256;

    int16_t lx = ((js->data[7] << 8) | js->data[6]);
    int16_t ly = ((js->data[9] << 8) | js->data[8]);
    int16_t rx = ((js->data[11] << 8) | js->data[10]);
    int16_t ry = ((js->data[13] << 8) | js->data[12]);

    input_report_abs(js->input, ABS_X, lx);
    input_report_abs(js->input, ABS_Y, -ly);
    input_report_abs(js->input, ABS_RY, rx);
    input_report_abs(js->input, ABS_RZ, -ry);
    input_report_abs(js->input, ABS_TILT_X, xx);
    input_report_abs(js->input, ABS_TILT_Y, yy);
    input_report_abs(js->input, ABS_RX, lt);
    input_report_abs(js->input, ABS_THROTTLE, rt);

    // ABS_RZ  ABS_RX     Rx    1001

    // 映射按钮状态
    int btn_a = js->data[3] & 0x10 ? 1 : 0;
    int btn_b = js->data[3] & 0x20 ? 1 : 0;
    int btn_x = js->data[3] & 0x40 ? 1 : 0;
    int btn_y = js->data[3] & 0x80 ? 1 : 0;
    int btn_lb = js->data[3] & 0x01 ? 1 : 0;
    int btn_rb = js->data[3] & 0x02 ? 1 : 0;
    int btn_home = js->data[3] & 0x04 ? 1 : 0;
    int btn_start = js->data[2] & 0x20 ? 1 : 0;
    int btn_meau = js->data[2] & 0x10 ? 1 : 0;
    int btn_tbl = js->data[2] & 0x40 ? 1 : 0;
    int btn_rbl = js->data[2] & 0x80 ? 1 : 0;

    input_report_key(js->input, BTN_A, btn_a);
    input_report_key(js->input, BTN_B, btn_b);
    input_report_key(js->input, BTN_X, btn_x);
    input_report_key(js->input, BTN_Y, btn_y);
    input_report_key(js->input, BTN_TL, btn_lb);
    input_report_key(js->input, BTN_TR, btn_rb);
    input_report_key(js->input, BTN_MODE, btn_home);
    input_report_key(js->input, BTN_START, btn_start);
    input_report_key(js->input, BTN_SELECT, btn_meau);
    input_report_key(js->input, BTN_THUMBL, btn_tbl);
    input_report_key(js->input, BTN_THUMBR, btn_rbl);

    // 确保事件同步
    input_sync(js->input);

    retval = usb_submit_urb(urb, GFP_ATOMIC);
    if (retval)
        dev_err(&js->udev->dev, "Failed to resubmit urb: %d\n", retval);
}

static int usb_joystick_open(struct input_dev *dev)
{
    struct usb_joystick *js = input_get_drvdata(dev);

    printk(KERN_INFO "usb_joystick_open: Opening joystick device\n");

    js->irq->dev = js->udev;
    if (usb_submit_urb(js->irq, GFP_KERNEL))
    {
        printk(KERN_ERR "usb_joystick_open: Failed to submit URB\n");
        return -EIO;
    }

    return 0;
}

static void usb_joystick_close(struct input_dev *dev)
{
    struct usb_joystick *js = input_get_drvdata(dev);

    printk(KERN_INFO "usb_joystick_close: Closing joystick device\n");

    usb_kill_urb(js->irq);
}

static int usb_joystick_probe(struct usb_interface *interface, const struct usb_device_id *id)
{
    struct usb_device *udev = interface_to_usbdev(interface);
    struct usb_endpoint_descriptor *endpoint;
    struct usb_joystick *js;
    struct input_dev *input_dev;
    int error;

    printk(KERN_INFO "usb_joystick_probe: Probing joystick device\n");

    js = kzalloc(sizeof(struct usb_joystick), GFP_KERNEL);
    if (!js)
        return -ENOMEM;

    js->udev = usb_get_dev(udev);

    endpoint = &interface->cur_altsetting->endpoint[0].desc;

    js->irq = usb_alloc_urb(0, GFP_KERNEL);
    if (!js->irq)
    {
        error = -ENOMEM;
        goto fail1;
    }

    js->data = usb_alloc_coherent(udev, BUFFER_SIZE, GFP_KERNEL, &js->data_dma);
    if (!js->data)
    {
        error = -ENOMEM;
        goto fail2;
    }

    usb_fill_int_urb(js->irq, udev,
                     usb_rcvintpipe(udev, endpoint->bEndpointAddress),
                     js->data, BUFFER_SIZE, usb_joystick_irq,
                     js, endpoint->bInterval);
    js->irq->transfer_dma = js->data_dma;
    js->irq->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;

    input_dev = input_allocate_device();
    if (!input_dev)
    {
        error = -ENOMEM;
        goto fail3;
    }

    js->input = input_dev;
    input_set_drvdata(input_dev, js);

    input_dev->name = "USB Joystick"; // 确保设备名称包含 "Joystick"
    input_dev->phys = "usb/js";
    input_dev->id.bustype = BUS_USB;
    input_dev->id.vendor = le16_to_cpu(udev->descriptor.idVendor);
    input_dev->id.product = le16_to_cpu(udev->descriptor.idProduct);
    input_dev->id.version = le16_to_cpu(udev->descriptor.bcdDevice);

    input_dev->dev.parent = &interface->dev;

    input_dev->open = usb_joystick_open;
    input_dev->close = usb_joystick_close;

    input_dev->evbit[0] = BIT_MASK(EV_KEY) | BIT_MASK(EV_ABS);
    input_dev->absbit[0] = BIT_MASK(ABS_X) | BIT_MASK(ABS_Y) | BIT_MASK(ABS_RX) | BIT_MASK(ABS_RZ) | BIT_MASK(ABS_TILT_X) |
                           BIT_MASK(ABS_TILT_Y) | BIT_MASK(ABS_RY) | BIT_MASK(ABS_THROTTLE);

    input_dev->keybit[BIT_WORD(BTN_A)] = BIT_MASK(BTN_A) | BIT_MASK(BTN_B) | BIT_MASK(BTN_X) | BIT_MASK(BTN_Y) |
                                         BIT_MASK(BTN_TL) | BIT_MASK(BTN_TR) | BIT_MASK(BTN_MODE) | BIT_MASK(BTN_START) |
                                         BIT_MASK(BTN_SELECT) | BIT_MASK(BTN_THUMBL) | BIT_MASK(BTN_THUMBR);


    input_set_abs_params(input_dev, ABS_X, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_Y, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_RX, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_RY, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_RZ, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_TILT_X, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_TILT_Y, -32768, 32767, 0, 0);
    input_set_abs_params(input_dev, ABS_THROTTLE, -32768, 32767, 0, 0);

    error = input_register_device(js->input);
    if (error)
        goto fail4;

    usb_set_intfdata(interface, js);
    return 0;

fail4:
    input_free_device(input_dev);
fail3:
    usb_free_coherent(udev, BUFFER_SIZE, js->data, js->data_dma);
fail2:
    usb_free_urb(js->irq);
fail1:
    kfree(js);
    usb_put_dev(udev);
    return error;
}

static void usb_joystick_disconnect(struct usb_interface *interface)
{
    struct usb_joystick *js = usb_get_intfdata(interface);

    printk(KERN_INFO "usb_joystick_disconnect: Disconnecting joystick device\n");

    usb_set_intfdata(interface, NULL);
    if (js)
    {
        usb_kill_urb(js->irq);
        input_unregister_device(js->input);
        usb_free_coherent(js->udev, BUFFER_SIZE, js->data, js->data_dma);
        usb_free_urb(js->irq);
        usb_put_dev(js->udev);
        kfree(js);
    }
}

static const struct usb_device_id usb_joystick_id_table[] = {
    {USB_DEVICE(USB_VENDOR_ID, USB_PRODUCT_ID)},
    {}};
MODULE_DEVICE_TABLE(usb, usb_joystick_id_table);

static struct usb_driver usb_joystick_driver = {
    .name = "usb_joystick",
    .probe = usb_joystick_probe,
    .disconnect = usb_joystick_disconnect,
    .id_table = usb_joystick_id_table,
};

module_usb_driver(usb_joystick_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Albusgive");
MODULE_DESCRIPTION("MOJIANG XBOX ONE USB");
