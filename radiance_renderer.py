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

fn uv_to_xytheta(uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    let length = 1.0 / f32(resolution);
    let x = u;
    let v_ = i32(floor(v * f32(resolution) * f32(angular_resolution) - 0.5));
    let theta_ = imod(v_, angular_resolution);
    let y_ = (v_ - theta_) / angular_resolution;

    let y = (f32(y_) + 0.5) / f32(resolution);
    let theta = (f32(theta_) + 0.5) / f32(angular_resolution);
    
    return vec3<f32>(x, y, theta);
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let xytheta = uv_to_xytheta(in.texCoord);
    let length = 1.0 / f32(resolution);
    let p = xytheta.xy;
    let theta = xytheta.z;
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

struct FragmentOutput {
    @location(0) color: vec4<f32>,
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

fn imod(x: i32, y: i32) -> i32 {
    let r = x % y;
    if (r < 0) {
        return r + y;
    }
    return r;
}

fn uv_to_xytheta(uv: vec2<f32>, res: i32, ares: i32) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    let length = 1.0 / f32(res);
    let x = u;
    let v_ = i32(floor(v * f32(res) * f32(ares) - 0.5));
    let theta_ = imod(v_, ares);
    let y_ = (v_ - theta_) / ares;

    let y = (f32(y_) + 0.5) / f32(res);
    let theta = (f32(theta_) + 0.5) / f32(ares);
    
    return vec3<f32>(x, y, theta);
}

fn sample_cascade(tex: texture_2d<f32>, smp: sampler, xytheta: vec3<f32>, res: i32, ares: i32) -> vec4<f32> {
    let x = xytheta.x;
    let y = xytheta.y;
    let theta = xytheta.z;

    let yf = y * f32(res) - 0.5;
    var yq = i32(floor(yf));
    let yr = yf - f32(yq);
    var yp = yq + 1;
    yq = clamp(yq, 0, res - 1);
    yp = clamp(yp, 0, res - 1);

    let af = theta * f32(ares) - 0.5;
    var aq = i32(floor(af));
    let ar = af - f32(aq);
    var ap = aq + 1;
    aq = imod(aq, ares);
    ap = imod(ap, ares);

    let rows = f32(ares * res);
    let v_qq = (f32(yq * ares + aq) + 0.5) / rows;
    let v_qp = (f32(yq * ares + ap) + 0.5) / rows;
    let v_pq = (f32(yp * ares + aq) + 0.5) / rows;
    let v_pp = (f32(yp * ares + ap) + 0.5) / rows;

    let c_qq = textureSample(tex, smp, vec2<f32>(x, v_qq));
    let c_qp = textureSample(tex, smp, vec2<f32>(x, v_qp));
    let c_pq = textureSample(tex, smp, vec2<f32>(x, v_pq));
    let c_pp = textureSample(tex, smp, vec2<f32>(x, v_pp));

    let c_q = c_qq * (1.0 - ar) + c_qp * ar;
    let c_p = c_pq * (1.0 - ar) + c_pp * ar;
    return c_q * (1.0 - yr) + c_p * yr;
}

override resolution: i32;
override angular_resolution: i32;
override ray_length: f32;


const PI: f32 = 3.14159265;


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let xytheta = uv_to_xytheta(in.texCoord, resolution, angular_resolution);
    let p = xytheta.xy;
    let theta = xytheta.z;
    let ray = vec2<f32>(cos(theta * 2 * PI), sin(theta * 2 * PI));
    let length = ray_length;
    let pa = p + ray * length;
    let pb = p + ray * 2 * length;

    out.color = sample_cascade(quadTexture, quadSampler, vec3<f32>(pa, theta), resolution * 2, angular_resolution / 2)
        + sample_cascade(quadTexture, quadSampler, vec3<f32>(pb, theta), resolution * 2, angular_resolution / 2);
    return out;
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

@group(0) @binding(0) var shortSampler: sampler;
@group(0) @binding(1) var shortTexture: texture_2d<f32>;
@group(0) @binding(2) var longSampler: sampler;
@group(0) @binding(3) var longTexture: texture_2d<f32>;


fn imod(x: i32, y: i32) -> i32 {
    let r = x % y;
    if (r < 0) {
        return r + y;
    }
    return r;
}

fn uv_to_xytheta(uv: vec2<f32>, res: i32, ares: i32) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    let length = 1.0 / f32(res);
    let x = u;
    let v_ = i32(floor(v * f32(res) * f32(ares) - 0.5));
    let theta_ = imod(v_, ares);
    let y_ = (v_ - theta_) / ares;

    let y = (f32(y_) + 0.5) / f32(res);
    let theta = (f32(theta_) + 0.5) / f32(ares);
    
    return vec3<f32>(x, y, theta);
}

fn sample_cascade(tex: texture_2d<f32>, smp: sampler, xytheta: vec3<f32>, res: i32, ares: i32) -> vec4<f32> {
    let x = xytheta.x;
    let y = xytheta.y;
    let theta = xytheta.z;

    let yf = y * f32(res) - 0.5;
    var yq = i32(floor(yf));
    let yr = yf - f32(yq);
    var yp = yq + 1;
    yq = clamp(yq, 0, res - 1);
    yp = clamp(yp, 0, res - 1);

    let af = theta * f32(ares) - 0.5;
    var aq = i32(floor(af));
    let ar = af - f32(aq);
    var ap = aq + 1;
    aq = imod(aq, ares);
    ap = imod(ap, ares);

    let rows = f32(ares * res);
    let v_qq = (f32(yq * ares + aq) + 0.5) / rows;
    let v_qp = (f32(yq * ares + ap) + 0.5) / rows;
    let v_pq = (f32(yp * ares + aq) + 0.5) / rows;
    let v_pp = (f32(yp * ares + ap) + 0.5) / rows;

    let c_qq = textureSample(tex, smp, vec2<f32>(x, v_qq));
    let c_qp = textureSample(tex, smp, vec2<f32>(x, v_qp));
    let c_pq = textureSample(tex, smp, vec2<f32>(x, v_pq));
    let c_pp = textureSample(tex, smp, vec2<f32>(x, v_pp));

    let c_q = c_qq * (1.0 - ar) + c_qp * ar;
    let c_p = c_pq * (1.0 - ar) + c_pp * ar;
    return c_q * (1.0 - yr) + c_p * yr;
}

override resolution: i32;
override angular_resolution: i32;
override ray_length: f32;

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

const PI: f32 = 3.14159265;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let short_uv = in.texCoord;
    let short_xytheta = uv_to_xytheta(short_uv, resolution, angular_resolution);

    out.color = textureSample(shortTexture, shortSampler, short_uv)
        + sample_cascade(longTexture, longSampler, short_xytheta, resolution / 2, angular_resolution * 2);
    return out;
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

fn xytheta_to_uv(xytheta: vec3<f32>) -> vec2<f32> {
    let x = xytheta.x;
    let y = xytheta.y;
    let theta = xytheta.z;
    
    let u = x;
    let theta_ = i32(floor(theta * f32(angular_resolution) - 0.5));
    let y_ = i32(floor(y * f32(resolution) - 0.5));
    let v_ = y_ * angular_resolution + theta_;
    let v = (f32(v_) + 0.5) / f32(angular_resolution * resolution);

    return vec2<f32>(u, v);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let x = in.texCoord[0];
    let y = in.texCoord[1];

    var color = vec4<f32>(0,0,0,0);
    for (var i = 0; i < 4; i++) {
        let theta = (f32(i) + 0.5) / 4.0;
        color += textureSample(quadTexture, quadSampler, xytheta_to_uv(vec3<f32>(x, y, theta)));
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
        self.merged_cascade_textures = []
        self.cascade_number = np.floor(np.log2(image.height)).astype(int)
        for i in range(self.cascade_number):
            self.cascades_textures.append(
                Texture(
                    image=image,
                    size=(image.height // (2**i) * 4 * (2**i), image.width // (2**i)),
                    format=wgpu.TextureFormat.rgba8unorm_srgb,
                    usage=wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.RENDER_ATTACHMENT,
                )
            )
            self.merged_cascade_textures.append(
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
        rad0_shader = device.create_shader_module(
            code=rad0_shader_source, label="rad0 shader source"
        )
        rad_shader = device.create_shader_module(
            code=rad_shader_source, label="rad shader source"
        )
        merge_shader = device.create_shader_module(
            code=merge_shader_source, label="merge shader"
        )
        collate_shader = device.create_shader_module(
            code=collate_shader_source, label="collate shader"
        )

        await self.light_texture.setup(setup_ctx)
        for cascade in self.cascades_textures:
            await cascade.setup(setup_ctx)

        for merge_cascade in self.merged_cascade_textures:
            await merge_cascade.setup(setup_ctx)

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

        self.merge_texture_bind_group_layout = (
            setup_ctx.device.create_bind_group_layout(
                label="radiance_renderer merge_texture_bind_group_layout",
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
                    wgpu.BindGroupLayoutEntry(
                        binding=2,
                        visibility=wgpu.ShaderStage.FRAGMENT,
                        sampler=wgpu.SamplerBindingLayout(),
                    ),
                    wgpu.BindGroupLayoutEntry(
                        binding=3,
                        visibility=wgpu.ShaderStage.FRAGMENT,
                        texture=wgpu.TextureBindingLayout(),
                    ),
                ],
            )
        )

        pipeline_layout = device.create_pipeline_layout(
            label="radiance_renderer pipeline_layout",
            bind_group_layouts=[
                texture_bind_group_layout,
            ],
        )
        merge_pipeline_layout = device.create_pipeline_layout(
            label="radiance_renderer merge_pipeline_layout",
            bind_group_layouts=[
                self.merge_texture_bind_group_layout,
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
        rad_pipelines = []
        for i in range(1, self.cascade_number):
            rad_pipelines.append(
                await device.create_render_pipeline_async(
                    label=f"radiance_renderer rad{i}_pipeline",
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
                        constants={
                            "resolution": self.cascades_textures[i].size[1],
                            "angular_resolution": self.cascades_textures[i].size[0]
                            // self.cascades_textures[i].size[1],
                            "ray_length": 2 ** (i - 1)
                            / self.cascades_textures[0].size[1],
                        },
                    ),
                    primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
                )
            )
        merge_pipelines = []
        for i in range(1, self.cascade_number):
            merge_pipelines.append(
                await device.create_render_pipeline_async(
                    label=f"radiance_renderer merge{i}_pipeline",
                    layout=merge_pipeline_layout,
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
                        constants={
                            "resolution": self.cascades_textures[i - 1].size[1],
                            "angular_resolution": self.cascades_textures[i - 1].size[0]
                            // self.cascades_textures[i - 1].size[1],
                            "ray_length": 2 ** (i - 1),
                        },
                    ),
                    primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
                )
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
        self.rad_pipelines = rad_pipelines
        self.merge_pipelines = merge_pipelines
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

        for n, cascade in enumerate(self.cascades_textures[1:]):

            cascade_pass: wgpu.GPURenderPassEncoder = (
                frame_ctx.command_encoder.begin_render_pass(
                    label=f"radiance_renderer_{n+1} render_pass",
                    color_attachments=[
                        wgpu.RenderPassColorAttachment(
                            view=cascade.texture.create_view(),
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

            radn_texture_bind_group = self.device.create_bind_group(
                label="radiance_renderer texture_bind_group",
                layout=self.cascades_texture_bind_group_layout,
                entries=[
                    wgpu.BindGroupEntry(binding=0, resource=sampler),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=self.cascades_textures[n].texture.create_view(),
                    ),
                ],
            )

            cascade_pass.set_pipeline(self.rad_pipelines[n])
            cascade_pass.set_bind_group(0, radn_texture_bind_group)
            cascade_pass.draw(6, 1, 0, 0)
            cascade_pass.end()

        self.merged_cascade_textures[-1] = self.cascades_textures[-1]
        for n in range(len(self.cascades_textures) - 1):
            rn = len(self.cascades_textures) - 1 - n
            merge_pass: wgpu.GPURenderPassEncoder = (
                frame_ctx.command_encoder.begin_render_pass(
                    label=f"radiance_renderer_{rn - 1} merge_pass",
                    color_attachments=[
                        wgpu.RenderPassColorAttachment(
                            view=self.merged_cascade_textures[
                                rn - 1
                            ].texture.create_view(),
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

            merge_texture_bind_group = self.device.create_bind_group(
                label="radiance_renderer merge_texture_bind_group",
                layout=self.merge_texture_bind_group_layout,
                entries=[
                    wgpu.BindGroupEntry(binding=0, resource=sampler),
                    wgpu.BindGroupEntry(
                        binding=1,
                        resource=self.cascades_textures[rn - 1].texture.create_view(),
                    ),
                    wgpu.BindGroupEntry(binding=2, resource=sampler),
                    wgpu.BindGroupEntry(
                        binding=3,
                        resource=self.merged_cascade_textures[rn].texture.create_view(),
                    ),
                ],
            )

            merge_pass.set_pipeline(self.merge_pipelines[rn - 1])
            merge_pass.set_bind_group(0, merge_texture_bind_group)
            merge_pass.draw(6, 1, 0, 0)
            merge_pass.end()

        collate_texture_bind_group = self.device.create_bind_group(
            label="radiance_renderer collate_texture_bind_group",
            layout=self.cascades_texture_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=sampler),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=self.merged_cascade_textures[0].texture.create_view(),
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
