import glfw
import wgpu
import sys
import os


class WgpuContext:
    def __init__(self, glfw_window):
        self.glfw_window = glfw_window
        self.wgpu_context = wgpu.gpu.get_canvas_context(
            self._get_glfw_present_methods()
        )
        self.wgpu_context.set_physical_size(
            *glfw.get_framebuffer_size(self.glfw_window)
        )

    def _get_glfw_present_methods(self):
        present_mthods_base = {
            "method": "screen",
            "vsync": False,
            "source": "WgpuContext",
        }
        if sys.platform.startswith("win"):
            return {
                "platform": "windows",
                "window": int(glfw.get_win32_window(self.glfw_window)),
                **present_mthods_base,
            }
        elif sys.platform.startswith("darwin"):
            return {
                "platform": "cocoa",
                "window": int(glfw.get_cocoa_window(self.glfw_window)),
                **present_mthods_base,
            }
        elif sys.platform.startswith("linux"):
            api_is_wayland = False
            system_is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if sys.platform.startswith("linux") and system_is_wayland:
                if not hasattr(glfw, "get_x11_window"):
                    api_is_wayland = True
            if api_is_wayland:
                return {
                    "platform": "wayland",
                    "window": int(glfw.get_wayland_window(self.glfw_window)),
                    "display": int(glfw.get_wayland_display()),
                    **present_mthods_base,
                }
            else:
                return {
                    "platform": "x11",
                    "window": int(glfw.get_x11_window(self.glfw_window)),
                    "display": int(glfw.get_x11_display()),
                    **present_mthods_base,
                }
        else:
            raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")

    def get_preferred_format(self, adapter):
        return self.wgpu_context.get_preferred_format(adapter)

    def configure(self, device, format):
        self.wgpu_context.configure(device=device, format=format)

    def unconfigure(self):
        self.wgpu_context.unconfigure()

    def get_current_texture(self):
        return self.wgpu_context.get_current_texture()

    def present(self):
        self.wgpu_context.present()
