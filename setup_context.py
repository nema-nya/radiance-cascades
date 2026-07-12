import dataclasses
import wgpu

from wgpu_context import WgpuContext


@dataclasses.dataclass
class SetupContext:
    wgpu_context: WgpuContext
    device: wgpu.GPUDevice
    render_texture_format: wgpu.enums.TextureFormatEnum
