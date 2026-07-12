import wgpu

from canvas import FrameContext
from setup_context import SetupContext
from texture import Texture


class BaseRenderer:
    def __init__(self, width, height):
        self.full_screen_texture = Texture(
            size=(height, width),
            format=wgpu.TextureFormat.bgra8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT,
            multisample=True,
            depth=True,
        )

    async def setup(self, setup_ctx: SetupContext):
        await self.full_screen_texture.setup(setup_ctx)

        self.context = setup_ctx.wgpu_context
        self.device = setup_ctx.device

    async def draw(self, frame_ctx: FrameContext):
        depth_stencil_attachment = None
        if self.full_screen_texture.depth:
            depth_stencil_attachment = wgpu.RenderPassDepthStencilAttachment(
                view=self.full_screen_texture.depth_target.create_view(),
                depth_load_op=wgpu.LoadOp.clear,
                depth_clear_value=1.0,
                depth_store_op=wgpu.StoreOp.store,
            )

        resolve_target = None
        if self.full_screen_texture.multisample:
            resolve_target = self.full_screen_texture.resolve_target.create_view()
        render_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="flat_renderer render_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.full_screen_texture.texture.create_view(),
                        resolve_target=resolve_target,
                        clear_value=(22 / 255, 22 / 255, 29 / 255, 1),
                        load_op="clear",
                        store_op="store",
                    )
                ],
                depth_stencil_attachment=depth_stencil_attachment,
            )
        )

        render_pass.end()
