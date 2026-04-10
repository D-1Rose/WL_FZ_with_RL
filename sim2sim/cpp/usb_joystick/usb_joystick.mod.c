#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xed386ed, "input_unregister_device" },
	{ 0x850f7662, "usb_free_coherent" },
	{ 0xb3e4b07e, "usb_free_urb" },
	{ 0x236c78bb, "usb_put_dev" },
	{ 0x37a0cba, "kfree" },
	{ 0x3e1c9c49, "usb_submit_urb" },
	{ 0x344ca289, "input_event" },
	{ 0x1e898af9, "_dev_err" },
	{ 0x1b613147, "usb_deregister" },
	{ 0x4c03a563, "random_kmalloc_seed" },
	{ 0xa63b4eed, "kmalloc_caches" },
	{ 0x59ffeca6, "kmalloc_trace" },
	{ 0xbf2b0c95, "usb_get_dev" },
	{ 0x1b06561d, "usb_alloc_urb" },
	{ 0x9e87fbdc, "usb_alloc_coherent" },
	{ 0x51c2b5f5, "input_allocate_device" },
	{ 0x3738402d, "input_set_abs_params" },
	{ 0xa9fdcc3f, "input_register_device" },
	{ 0x436ecf65, "input_free_device" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x97c4004d, "usb_register_driver" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x122c3a7e, "_printk" },
	{ 0xc2ca6115, "usb_kill_urb" },
	{ 0xf079b8f9, "module_layout" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("usb:v413Dp2104d*dc*dsc*dp*ic*isc*ip*in*");

MODULE_INFO(srcversion, "84E5FDCD1F1F8A72C357B57");
