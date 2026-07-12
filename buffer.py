import wgpu
import numpy as np

from setup_context import SetupContext


class Buffer:
    def __init__(
        self,
        data=None,
        usage=wgpu.BufferUsage.UNIFORM,
        shape=None,
        dtype=None,
        staging=False,
    ):
        if data is not None:
            assert shape is None or shape == data.shape
            assert dtype is None or dtype == data.dtype
        if shape is None:
            assert data is not None
            shape = data.shape
        if dtype is None:
            assert data is not None
            dtype = data.dtype
        dtype = np.dtype(dtype)

        self.data = data
        self.shape = shape
        self.nbytes = np.prod(shape) * dtype.itemsize
        self.dtype = dtype
        self.usage = usage
        self.staging = staging

    async def setup(self, setup_ctx: SetupContext):
        self.buffer = setup_ctx.device.create_buffer(
            size=self.nbytes,
            usage=self.usage + wgpu.BufferUsage.COPY_DST,
        )
        if self.staging:
            self.staging_buffer = setup_ctx.device.create_buffer(
                size=self.nbytes,
                usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.MAP_WRITE,
            )

    async def bind(
        self, device: wgpu.GPUDevice, command_encoder: wgpu.GPUCommandEncoder
    ):
        if self.data is not None:
            if not self.staging:
                staging = device.create_buffer_with_data(
                    data=self.data, usage=wgpu.BufferUsage.COPY_SRC
                )
            else:
                await self.staging_buffer.map_async(
                    mode=wgpu.MapMode.WRITE, offset=0, size=self.nbytes
                )
                self.staging_buffer.write_mapped(self.data, buffer_offset=0)
                self.staging_buffer.unmap()
                staging = self.staging_buffer
            command_encoder.copy_buffer_to_buffer(
                source=staging,
                source_offset=0,
                destination=self.buffer,
                destination_offset=0,
                size=np.prod(self.shape) * self.dtype.itemsize,
            )
            self.data = None
        return self.buffer

    async def schedule_load(self, data):
        assert data.shape == self.shape
        assert data.dtype == self.dtype
        self.data = data
