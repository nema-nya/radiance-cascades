import numpy as np
import wgpu

from canvas import FrameContext
from setup_context import SetupContext
from buffer import Buffer
from texture import Texture
from camera import Camera

flat_shader_source = """

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;

@group(1) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> tex_coords: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read> normals: array<vec4<f32>>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = mvp * positions[in.vertex_index];
    out.tex_coord = tex_coords[in.vertex_index];
    out.normal = normals[in.vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.normal.rgb * vec3<f32>(0.5) + vec3<f32>(0.5), 1.0);
}
"""


class ModelRenderer:
    def __init__(self, model, target: Texture, camera: Camera):
        self.mvp_buffer = Buffer(shape=(4, 4), dtype=np.float32, staging=True)
        self.model = model
        self.positions_buffer = Buffer(
            data=model.positions,
            usage=wgpu.BufferUsage.STORAGE,
        )
        self.tex_coords_buffer = Buffer(
            data=model.tex_coords,
            usage=wgpu.BufferUsage.STORAGE,
        )
        self.normals_buffer = Buffer(
            data=model.normals,
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.target = target
        self.camera = camera

    async def setup(self, setup_ctx: SetupContext):
        shader = setup_ctx.device.create_shader_module(code=flat_shader_source)

        await self.mvp_buffer.setup(setup_ctx)
        await self.positions_buffer.setup(setup_ctx)
        await self.tex_coords_buffer.setup(setup_ctx)
        await self.normals_buffer.setup(setup_ctx)

        uniform_bind_group_layout = setup_ctx.device.create_bind_group_layout(
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.VERTEX,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.uniform
                    ),
                )
            ]
        )

        vertices_bind_group_layout = setup_ctx.device.create_bind_group_layout(
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.VERTEX,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.VERTEX,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.VERTEX,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage
                    ),
                ),
            ]
        )

        pipeline_layout = setup_ctx.device.create_pipeline_layout(
            bind_group_layouts=[
                uniform_bind_group_layout,
                vertices_bind_group_layout,
            ]
        )

        depth_stencil = None
        if self.target.depth:
            depth_stencil = wgpu.DepthStencilState(
                format=wgpu.TextureFormat.depth32float,
                depth_write_enabled=True,
                depth_compare=wgpu.CompareFunction.less,
            )
        multisample = None
        if self.target.multisample:
            multisample = wgpu.MultisampleState(count=4)

        render_pipeline = await setup_ctx.device.create_render_pipeline_async(
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=shader,
                entry_point="vs_main",
            ),
            depth_stencil=depth_stencil,
            multisample=multisample,
            fragment=wgpu.FragmentState(
                module=shader,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=self.target.format,
                        blend={"color": {}, "alpha": {}},
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(cull_mode=wgpu.CullMode.back),
        )

        self.uniform_bind_group_layout = uniform_bind_group_layout
        self.vertices_bind_group_layout = vertices_bind_group_layout
        self.context = setup_ctx.wgpu_context
        self.device = setup_ctx.device
        self.render_pipeline = render_pipeline

    async def draw(self, frame_ctx: FrameContext):
        model_mat = self.model.transform
        view_mat = self.camera.view_mat()
        proj_mat = self.camera.projection_mat()
        mvp = proj_mat @ view_mat @ model_mat
        await self.mvp_buffer.schedule_load(np.ascontiguousarray(mvp.T))

        mvp_buffer = await self.mvp_buffer.bind(self.device, frame_ctx.command_encoder)

        uniform_bind_group = self.device.create_bind_group(
            layout=self.uniform_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0, resource=wgpu.BufferBinding(buffer=mvp_buffer)
                )
            ],
        )

        positions_buffer = await self.positions_buffer.bind(
            self.device, frame_ctx.command_encoder
        )
        tex_coords_buffer = await self.tex_coords_buffer.bind(
            self.device, frame_ctx.command_encoder
        )
        normals_buffer = await self.normals_buffer.bind(
            self.device, frame_ctx.command_encoder
        )

        vertices_bind_group = self.device.create_bind_group(
            label="model_renderer vertices_bind_group",
            layout=self.vertices_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(binding=0, resource=positions_buffer),
                wgpu.BindGroupEntry(binding=1, resource=tex_coords_buffer),
                wgpu.BindGroupEntry(binding=2, resource=normals_buffer),
            ],
        )

        depth_stencil_attachment = None
        if self.target.depth:
            depth_stencil_attachment = wgpu.RenderPassDepthStencilAttachment(
                view=self.target.depth_target.create_view(),
                depth_load_op=wgpu.LoadOp.load,
                depth_store_op=wgpu.StoreOp.store,
            )

        resolve_target = None
        if self.target.multisample:
            resolve_target = self.target.resolve_target.create_view()
        render_pass: wgpu.GPURenderPassEncoder = (
            frame_ctx.command_encoder.begin_render_pass(
                label="model_renderer render_pass",
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=self.target.texture.create_view(),
                        resolve_target=resolve_target,
                        load_op=wgpu.LoadOp.load,
                        store_op=wgpu.StoreOp.store,
                    )
                ],
                depth_stencil_attachment=depth_stencil_attachment,
            )
        )

        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, uniform_bind_group)
        render_pass.set_bind_group(1, vertices_bind_group)
        render_pass.draw(self.positions_buffer.shape[0], 1, 0, 0)
        render_pass.end()
