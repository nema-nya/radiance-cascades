import asyncio

import numpy as np
import glfw

from composer_renderer import ComposerRenderer
from canvas import Canvas
from sequential_renderer import SequentialRenderer
from camera import Camera
from base_renderer import BaseRenderer


class Controller:
    def __init__(self, camera: Camera):
        self.should_close = False
        self.yaw_angle = 0.0
        self.left_pressed = False
        self.last_cursor_pos = None
        self.camera = camera

    def on_framebuffer_size(self, *_):
        pass

    def on_key(self, *args):
        key, action = args[1], args[3]
        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.should_close = True

    def _sync_camera(self):
        a = self.yaw_angle / 1000.0 * 10.0
        camera_pos = np.array([np.sin(a), 0.0, -np.cos(a)]) * 5.0
        self.camera.position = camera_pos

    def on_cursor_pos(self, *args):
        x, y = args[1], args[2]
        if self.left_pressed:
            assert self.last_cursor_pos is not None
            self.yaw_angle -= x - self.last_cursor_pos[0]
            self._sync_camera()
            self.last_cursor_pos = (x, y)

    def on_mouse_button(self, *args):
        window, key, action = args[0], args[1], args[2]
        if key == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.left_pressed = True
                self.last_cursor_pos = glfw.get_cursor_pos(window)
            else:
                self.left_pressed = False

    def on_setup(self, *_):
        pass

    def on_draw(self, *_):
        if not self.left_pressed:
            self.yaw_angle += 1.0
            self._sync_camera()


async def main():
    w, h = 480, 720
    camera = Camera(
        position=np.array([0.0, 0.0, -5]),
        target=np.array([0.0, 0.0, 0.0]),
        fov=90.0 / 360.0 * 2.0 * np.pi,
        aspect=w / h,
        near=1.0,
        far=10.0,
    )

    base_renderer = BaseRenderer(w, h)
    composer_renderer = ComposerRenderer(base_renderer.full_screen_texture)
    controller = Controller(camera)

    canvas = Canvas(
        SequentialRenderer(
            [
                base_renderer,
                composer_renderer,
            ]
        ),
        width=w,
        height=h,
        title="Showroom",
        controller=controller,
    )
    await canvas.run()


if __name__ == "__main__":
    asyncio.run(main())
