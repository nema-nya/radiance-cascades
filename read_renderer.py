import PIL
import numpy as np
import wgpu

from canvas import FrameContext
from setup_context import SetupContext
from buffer import Buffer
from texture import Texture

quad_shader_source = """
@group(0) @binding(0) var<uniform> gamma: f32;

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct VertexOutput {
    @location(0) texCoord : vec2<f32>,
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    var texCoords = array<vec2<f32>, 6>(  // srgb colors
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    let index = i32(in.vertex_index);
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[index], 0.0, 1.0);
    out.texCoord = texCoords[index];
    return out;
}

@group(1) @binding(0) var quadSampler: sampler;
@group(1) @binding(1) var quadTexture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(quadTexture, quadSampler, in.texCoord);
    let physical_color = vec4<f32>(pow(tex_color.rgb, vec3<f32>(gamma)), tex_color.a);
    return physical_color;
}
"""


class ReadRenderer:
    def __init__(self, texture):
        self.texture = texture
        self.gamma = 2.2
        self.gamma_buffer = Buffer(np.array([self.gamma], dtype=np.float32))
        self.staging_target = Texture(
            size=(self.texture.size[0], self.texture.size[1]),
            format=wgpu.TextureFormat.bgra8unorm_srgb,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        h = self.texture.size[0]
        w = self.texture.size[1]
        number_of_bytes_per_row = w * 4
        number_of_bytes_per_row_padded = ((number_of_bytes_per_row + 255) // 256) * 256
        self.staging_buffer = Buffer(
            data=None,
            usage=wgpu.BufferUsage.MAP_READ,
            shape=np.array([h, number_of_bytes_per_row_padded]),
            dtype=np.uint8,
            staging=False,
        )
        self.saved = False
        self.i = 0

    async def setup(self, setup_ctx: SetupContext):
        device = setup_ctx.device
        shader = device.create_shader_module(code=quad_shader_source)

        await self.gamma_buffer.setup(setup_ctx)
        await self.staging_target.setup(setup_ctx)
        await self.staging_buffer.setup(setup_ctx)

        uniform_bind_group_layout = device.create_bind_group_layout(
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    buffer=wgpu.BufferBindingLayout(),
                ),
            ]
        )

        texture_bind_group_layout = setup_ctx.device.create_bind_group_layout(
            label="read_renderer texture_bind_group_layout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    sampler=wgpu.SamplerBindingLayout(),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    texture=wgpu.TextureBindingLayout(),
                ),
            ],
        )

        pipeline_layout = device.create_pipeline_layout(
            label="read_renderer pipeline_layout",
            bind_group_layouts=[
                uniform_bind_group_layout,
                texture_bind_group_layout,
            ],
        )

        render_pipeline = await device.create_render_pipeline_async(
            label="read_renderer render_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=None,
            fragment=wgpu.FragmentState(
                module=shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=setup_ctx.render_texture_format,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )

        self.uniform_bind_group_layout = uniform_bind_group_layout
        self.texture_bind_group_layout = texture_bind_group_layout
        self.context = setup_ctx.wgpu_context
        self.device = device
        self.render_pipeline = render_pipeline

    async def draw(self, frame_ctx: FrameContext):
        gamma_buffer = await self.gamma_buffer.bind(
            self.device, frame_ctx.command_encoder
        )

        uniform_bind_group = self.device.create_bind_group(
            layout=self.uniform_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0, resource=wgpu.BufferBinding(buffer=gamma_buffer)
                ),
            ],
        )

        await self.texture.bind(self.device, frame_ctx.command_encoder)

        if self.texture.multisample:
            texture = self.texture.resolve_target.create_view()
        else:
            texture = self.texture.texture.create_view()

        sampler = self.device.create_sampler(
            label="read_renderer sampler",
            min_filter=wgpu.FilterMode.nearest,
            mag_filter=wgpu.FilterMode.nearest,
        )

        texture_bind_group = self.device.create_bind_group(
            label="read_renderer texture_bind_group",
            layout=self.texture_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=sampler),
                wgpu.BindGroupEntry(binding=1, resource=texture),
            ],
        )
        resolve_target = None
        if self.staging_target.multisample:
            resolve_target = self.staging_target.resolve_target.create_view()
        render_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="read_renderer render_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.staging_target.texture.create_view(),
                        resolve_target=resolve_target,
                        clear_value=(
                            0,
                            0,
                            0,
                            1,
                        ),
                        load_op="clear",
                        store_op="store",
                    )
                ],
            )
        )

        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, uniform_bind_group)
        render_pass.set_bind_group(1, texture_bind_group)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()
        await self.staging_buffer.bind(self.device, frame_ctx.command_encoder)
        # self.staging_target.texture.
        frame_ctx.command_encoder.copy_texture_to_buffer(
            {"texture": self.staging_target.texture},
            {
                "buffer": self.staging_buffer.buffer,
                "offset": 0,
                "bytes_per_row": self.staging_buffer.shape[1],
                "rows_per_image": self.staging_buffer.shape[0],
            },
            (self.texture.size[0], self.texture.size[1], 1),
        )
        await self.staging_buffer.buffer.map_async(wgpu.MapMode.READ)
        image = self.staging_buffer.buffer.read_mapped()
        self.staging_buffer.buffer.unmap()
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.staging_buffer.shape)
        image = image[:, : self.texture.size[1] * 4]
        image = image.reshape(self.texture.size[0], self.texture.size[1], 4)
        image = PIL.Image.fromarray(image, mode="RGBA")

        self.i += 1
        if self.i == 2:
            image.save("out.png")
