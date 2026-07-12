import dataclasses
import wgpu


@dataclasses.dataclass
class FrameContext:
    fps: float | None
    frame_index: int
    size: tuple[int, int]
    command_encoder: wgpu.GPUCommandEncoder
