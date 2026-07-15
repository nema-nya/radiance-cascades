import numpy as np
import wgpu

from canvas import FrameContext
from setup_context import SetupContext
from buffer import Buffer
from texture import Texture

rad0_shader_source = """

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct VertexOutput {
    @location(0) texCoord : vec2<f32>,
    @builtin(position) pos: vec4<f32>,
};

@group(0) @binding(0) var quadSampler: sampler;
@group(0) @binding(1) var quadTexture: texture_2d<f32>;

override resolution: i32;
override angular_resolution: i32;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

const PI: f32 = 3.14159265;

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

fn imod(x: i32, y: i32) -> i32 {
    let r = x % y;
    if (r < 0) {
        return r + y;
    }
    return r;
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let u = in.texCoord[0];
    let v = in.texCoord[1];
    let length = 1.0 / f32(resolution);
    let x = u;
    let v_ = i32(floor(v * f32(resolution) * f32(angular_resolution) - 0.5));
    let theta_ = imod(v_, angular_resolution);
    let y_ = (v_ - theta_) / angular_resolution;

    let y = (f32(y_) + 0.5) / f32(resolution);
    let theta = (f32(theta_) + 0.5) / f32(angular_resolution) ;

    let p = vec2<f32>(x, y);
    let ray = vec2<f32>(cos(theta * 2 * PI), sin(theta * 2 * PI));
    let q = p + ray * length;
    out.color = textureSample(quadTexture, quadSampler, q);
    return out;
}
"""

rad_shader_source = """

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
    //let tex_color = textureSample(quadTexture, quadSampler, in.texCoord);
    let tex_color = vec4<f32>(in.texCoord.xy, 0.0, 1.0);
    return tex_color;
}
"""


merge_shader_source = """

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
    //let tex_color = textureSample(quadTexture, quadSampler, in.texCoord);
    let tex_color = vec4<f32>(in.texCoord.xy, 0.0, 1.0);
    return tex_color;
}
"""


collate_shader_source = """

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

override resolution: i32;
override angular_resolution: i32;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

fn imod(x: i32, y: i32) -> i32 {
    let r = x % y;
    if (r < 0) {
        return r + y;
    }
    return r;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let x = in.texCoord[0];
    let y = in.texCoord[1];

    var color = vec4<f32>(0,0,0,0);
    for (var i = 0; i < 4; i++) {
        let theta = (f32(i) + 0.5) / 4.0;
        let u = x;
        let theta_ = i;
        let y_ = i32(floor(y * f32(resolution) - 0.5));
        let v_ = y_ * angular_resolution + theta_;
        let v = (f32(v_) + 0.5) / f32(angular_resolution * resolution);
        color += textureSample(quadTexture, quadSampler, vec2<f32>(u, v));
    }
    out.color = color / 4.0;
    return out;
}
"""


class RadianceRenderer:
    def __init__(self, image, target):
        self.image = image
        self.target = target
        self.light_texture = Texture(
            image=image,
            size=(image.height, image.width),
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.cascades_textures = []
        cascade_number = np.floor(np.log2(image.height)).astype(int)
        for i in range(cascade_number):
            self.cascades_textures.append(
                Texture(
                    image=image,
                    size=(image.height // (2**i) * 4 * (2**i), image.width // (2**i)),
                    format=wgpu.TextureFormat.rgba8unorm_srgb,
                    usage=wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.RENDER_ATTACHMENT,
                )
            )

    async def setup(self, setup_ctx: SetupContext):
        device = setup_ctx.device
        rad0_shader = device.create_shader_module(code=rad0_shader_source)
        rad_shader = device.create_shader_module(code=rad_shader_source)
        merge_shader = device.create_shader_module(code=merge_shader_source)
        collate_shader = device.create_shader_module(code=collate_shader_source)

        await self.light_texture.setup(setup_ctx)
        for cascade in self.cascades_textures:
            await cascade.setup(setup_ctx)

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

        rad0_pipeline = await device.create_render_pipeline_async(
            label="radiance_renderer rad0_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=rad0_shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=None,
            fragment=wgpu.FragmentState(
                module=rad0_shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=wgpu.TextureFormat.rgba8unorm_srgb,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
                constants={
                    "resolution": self.cascades_textures[0].size[1],
                    "angular_resolution": self.cascades_textures[0].size[0]
                    // self.cascades_textures[0].size[1],
                },
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )
        print(self.cascades_textures[0].size)
        rad_pipeline = await device.create_render_pipeline_async(
            label="radiance_renderer rad_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=rad_shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=None,
            fragment=wgpu.FragmentState(
                module=rad_shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=wgpu.TextureFormat.rgba8unorm_srgb,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )

        merge_pipeline = await device.create_render_pipeline_async(
            label="radiance_renderer merge_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=merge_shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=None,
            fragment=wgpu.FragmentState(
                module=merge_shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=wgpu.TextureFormat.rgba8unorm_srgb,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )
        multisample = None
        if self.target.multisample:
            multisample = wgpu.MultisampleState(count=4)
        collate_pipeline = await device.create_render_pipeline_async(
            label="radiance_renderer collate_pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=merge_shader,
                entry_point="vs_main",
            ),
            depth_stencil=None,
            multisample=multisample,
            fragment=wgpu.FragmentState(
                module=collate_shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=self.target.format,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
                constants={
                    "resolution": self.cascades_textures[0].size[1],
                    "angular_resolution": self.cascades_textures[0].size[0]
                    // self.cascades_textures[0].size[1],
                },
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )
        self.cascades_texture_bind_group_layout = texture_bind_group_layout
        self.context = setup_ctx.wgpu_context
        self.device = device
        self.rad0_pipeline = rad0_pipeline
        self.rad_pipeline = rad_pipeline
        self.merge_pipeline = merge_pipeline
        self.collate_pipeline = collate_pipeline

    async def draw(self, frame_ctx: FrameContext):
        await self.light_texture.bind(self.device, frame_ctx.command_encoder)

        sampler = self.device.create_sampler(
            label="radiance_renderer sampler",
            min_filter=wgpu.FilterMode.linear,
            mag_filter=wgpu.FilterMode.linear,
        )

        rad0_texture_bind_group = self.device.create_bind_group(
            label="radiance_renderer rad0_texture_bind_group",
            layout=self.cascades_texture_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=sampler),
                wgpu.BindGroupEntry(
                    binding=1, resource=self.light_texture.texture.create_view()
                ),
            ],
        )
        rad0_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="radiance_renderer rad0_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.cascades_textures[0].texture.create_view(),
                        resolve_target=None,
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

        rad0_pass.set_pipeline(self.rad0_pipeline)
        rad0_pass.set_bind_group(0, rad0_texture_bind_group)
        rad0_pass.draw(6, 1, 0, 0)
        rad0_pass.end()

        # for n, cascade in enumerate(self.cascades_textures[1:]):

        #     cascade_pass: wgpu.GPURenderPassEncoder = (
        #         frame_ctx.command_encoder.begin_render_pass(
        #             label=f"radiance_renderer_{n} render_pass",
        #             color_attachments=[
        #                 wgpu.RenderPassColorAttachment(
        #                     view=cascade.texture.create_view(),
        #                     resolve_target=None,
        #                     clear_value=(
        #                         0,
        #                         0,
        #                         0,
        #                         1,
        #                     ),
        #                     load_op="clear",
        #                     store_op="store",
        #                 )
        #             ],
        #         )
        #     )

        #     radn_texture_bind_group = self.device.create_bind_group(
        #         label="radiance_renderer texture_bind_group",
        #         layout=self.cascades_texture_bind_group_layout,
        #         entries=[
        #             wgpu.BindGroupEntry(binding=0, resource=sampler),
        #             wgpu.BindGroupEntry(
        #                 binding=1,
        #                 resource=self.cascades_textures[n - 1].texture.create_view(),
        #             ),
        #         ],
        #     )

        #     cascade_pass.set_pipeline(self.rad_pipeline)
        #     cascade_pass.set_bind_group(0, radn_texture_bind_group)
        #     cascade_pass.draw(6, 1, 0, 0)
        #     cascade_pass.end()

        # for n in range(len(self.cascades_textures) - 1):
        #     rn = len(self.cascades_textures) - 1 - n
        #     merge_pass: wgpu.GPURenderPassEncoder = (
        #         frame_ctx.command_encoder.begin_render_pass(
        #             label=f"radiance_renderer_{n} merge_pass",
        #             color_attachments=[
        #                 wgpu.RenderPassColorAttachment(
        #                     view=self.cascades_textures[rn - 1].texture.create_view(),
        #                     resolve_target=None,
        #                     clear_value=(
        #                         0,
        #                         0,
        #                         0,
        #                         1,
        #                     ),
        #                     load_op="clear",
        #                     store_op="store",
        #                 )
        #             ],
        #         )
        #     )

        #     merge_texture_bind_group = self.device.create_bind_group(
        #         label="radiance_renderer merge_texture_bind_group",
        #         layout=self.cascades_texture_bind_group_layout,
        #         entries=[
        #             wgpu.BindGroupEntry(binding=0, resource=sampler),
        #             wgpu.BindGroupEntry(
        #                 binding=1,
        #                 resource=self.cascades_textures[rn].texture.create_view(),
        #             ),
        #         ],
        #     )

        #     merge_pass.set_pipeline(self.merge_pipeline)
        #     merge_pass.set_bind_group(0, merge_texture_bind_group)
        #     merge_pass.draw(6, 1, 0, 0)
        #     merge_pass.end()

        collate_texture_bind_group = self.device.create_bind_group(
            label="radiance_renderer collate_texture_bind_group",
            layout=self.cascades_texture_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=sampler),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self.cascades_textures[0].texture.create_view(),
                ),
            ],
        )
        resolve_target = None
        if self.target.multisample:
            resolve_target = self.target.resolve_target.create_view()
        collate_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="radiance_renderer collate_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.target.texture.create_view(),
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

        collate_pass.set_pipeline(self.collate_pipeline)
        collate_pass.set_bind_group(0, collate_texture_bind_group)
        collate_pass.draw(6, 1, 0, 0)
        collate_pass.end()
