import numpy as np
import wgpu

from canvas import FrameContext
from setup_context import SetupContext
from buffer import Buffer
from texture import Texture

quad_shader_source = """

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

@group(0) @binding(0) var quadSampler: sampler;
@group(0) @binding(1) var quadTexture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(quadTexture, quadSampler, in.texCoord);
    //let tex_color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return tex_color;
}
"""


class RadianceRenderer:
    def __init__(self, image, texture):
        self.image = image
        self.texture = texture
        self.light_texture = Texture(
            image=image,
            size=(image.height, image.width),
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING,
        )

    async def setup(self, setup_ctx: SetupContext):
        device = setup_ctx.device
        shader = device.create_shader_module(code=quad_shader_source)

        await self.light_texture.setup(setup_ctx)

        texture_bind_group_layout = setup_ctx.device.create_bind_group_layout(
            label="radiance_renderer texture_bind_group_layout",
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
            label="radiance_renderer pipeline_layout",
            bind_group_layouts=[
                texture_bind_group_layout,
            ],
        )

        multisample = None
        if self.texture.multisample:
            multisample = wgpu.MultisampleState(count=4)

        render_pipeline = await device.create_render_pipeline_async(
            label="radiance_renderer render_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=multisample,
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
        self.texture_bind_group_layout = texture_bind_group_layout
        self.context = setup_ctx.wgpu_context
        self.device = device
        self.render_pipeline = render_pipeline

    async def draw(self, frame_ctx: FrameContext):
        await self.light_texture.bind(self.device, frame_ctx.command_encoder)

        texture = self.light_texture.texture.create_view()

        sampler = self.device.create_sampler(
            label="radiance_renderer sampler",
            min_filter=wgpu.FilterMode.nearest,
            mag_filter=wgpu.FilterMode.nearest,
        )

        texture_bind_group = self.device.create_bind_group(
            label="radiance_renderer texture_bind_group",
            layout=self.texture_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=sampler),
                wgpu.BindGroupEntry(binding=1, resource=texture),
            ],
        )
        resolve_target = None
        if self.texture.multisample:
            resolve_target = self.texture.resolve_target.create_view()
        render_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="radiance_renderer render_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.texture.texture.create_view(),
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
        render_pass.set_bind_group(0, texture_bind_group)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()
