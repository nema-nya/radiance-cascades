import wgpu
import glfw
import time

from setup_context import SetupContext
from wgpu_context import WgpuContext
from frame_context import FrameContext


class Canvas:
    def __init__(self, renderer, height, width, title="", controller=None):
        if not glfw.init():
            return

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.VISIBLE, False)  # start hidden

        window = glfw.create_window(width, height, title, None, None)
        if not window:
            glfw.terminate()
            return

        glfw.set_framebuffer_size_callback(window, self.on_framebuffer_size)
        if controller is not None:
            glfw.set_key_callback(window, controller.on_key)
            glfw.set_mouse_button_callback(window, controller.on_mouse_button)
            glfw.set_cursor_pos_callback(window, controller.on_cursor_pos)

        context = WgpuContext(window)

        self.renderer = renderer
        self.controller = controller
        self.window = window
        self.context = context

        glfw.show_window(window)

    def on_framebuffer_size(self, window, w, h):
        self.context.wgpu_context.set_physical_size(w, h)
        if self.controller is not None:
            self.controller.on_framebuffer_size(window, w, h)

    async def run(self):
        adapter = await wgpu.gpu.request_adapter_async(
            power_preference="high-performance"
        )
        device: wgpu.GPUDevice = await adapter.request_device_async()

        render_texture_format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=render_texture_format)

        setup_ctx = SetupContext(self.context, device, render_texture_format)
        if self.controller is not None:
            self.controller.on_setup(setup_ctx)
        await self.renderer.setup(setup_ctx)

        prev_time = None
        frame_deltas = []
        frame_index = 0
        while not glfw.window_should_close(self.window):
            frame_time = time.time()
            fps = None
            if prev_time is not None:
                frame_delta = frame_time - prev_time
                frame_deltas.append(frame_delta)
                frame_deltas = frame_deltas[-128:]
                avg_frame_delta = sum(frame_deltas) / len(frame_deltas)
                fps = 1 / avg_frame_delta
            prev_time = frame_time

            command_encoder = device.create_command_encoder()

            frame_ctx = FrameContext(
                fps=fps,
                frame_index=frame_index,
                size=glfw.get_framebuffer_size(self.window),
                command_encoder=command_encoder,
            )

            if self.controller is not None:
                self.controller.on_draw(frame_ctx)
            await self.renderer.draw(frame_ctx)

            device.queue.submit([command_encoder.finish()])

            self.context.present()

            glfw.poll_events()
            if self.controller is not None and self.controller.should_close:
                break
            frame_index += 1

        self.context.unconfigure()
        glfw.destroy_window(self.window)
        glfw.terminate()
