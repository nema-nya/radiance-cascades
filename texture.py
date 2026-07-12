import wgpu
import numpy as np
import PIL.Image

from setup_context import SetupContext


class Texture:
    def __init__(
        self,
        image=None,
        label="",
        size: tuple[int, int] | None = None,
        usage=wgpu.TextureUsage.TEXTURE_BINDING,
        format=wgpu.enums.TextureFormat.rgba8unorm_srgb,
        multisample=False,
        depth=False,
    ):
        if image is not None:
            image = np.array(image)
        if size is None:
            assert image is not None
            size = image.shape[:2]
        self.label = label
        self.size = size
        self.image = image
        self.multisample = multisample
        self.depth = depth
        self.format = format
        self.usage = usage

    async def setup(self, setup_ctx: SetupContext):
        self.texture = setup_ctx.device.create_texture(
            label=self.label,
            size=(self.size[1], self.size[0], 1),
            sample_count=(4 if self.multisample else 1),
            format=self.format,
            usage=self.usage + wgpu.TextureUsage.COPY_DST,
        )
        if self.multisample:
            self.resolve_target = setup_ctx.device.create_texture(
                label=f"{self.label}-resolve_target",
                size=(self.size[1], self.size[0], 1),
                sample_count=1,
                format=self.format,
                usage=wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
        if self.depth:
            self.depth_target = setup_ctx.device.create_texture(
                label=f"{self.label}-depth_target",
                size=(self.size[1], self.size[0], 1),
                sample_count=(4 if self.multisample else 1),
                format=wgpu.TextureFormat.depth32float,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
            )

    async def bind(self, device, command_encoder: wgpu.GPUCommandEncoder):
        if self.image is None:
            return
        data_rgba = self.image
        if type(data_rgba) is PIL.Image:
            data_rgba = np.array(data_rgba)

        data_rgba = data_rgba.reshape(self.size[0], self.size[1] * 4)
        texture_width_bytes = self.size[1] * 4
        texture_width_bytes_aligned = int((texture_width_bytes + 255) / 256) * 256
        data_aligned = np.zeros(
            (self.size[0], texture_width_bytes_aligned), dtype=np.uint8
        )
        data_aligned[:, :texture_width_bytes] = data_rgba
        data_aligned = data_aligned.flatten()

        texture_buffer = device.create_buffer_with_data(
            data=data_aligned, usage=wgpu.BufferUsage.COPY_SRC
        )
        command_encoder.copy_buffer_to_texture(
            source=wgpu.TexelCopyBufferInfo(
                bytes_per_row=texture_width_bytes_aligned,
                rows_per_image=self.size[0],
                buffer=texture_buffer,
            ),
            destination=wgpu.TexelCopyTextureInfo(texture=self.texture),
            copy_size=(self.size[1], self.size[0], 1),
        )
        self.image = None
