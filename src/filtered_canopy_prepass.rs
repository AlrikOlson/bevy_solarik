//! Publish finite first events before the main pass, including temporal guides.
use super::CanopyGpu;
use bevy_asset::{AssetServer, load_embedded_asset};
use bevy_core_pipeline::{
    FullscreenShader,
    core_3d::CORE_3D_DEPTH_FORMAT,
    deferred::{
        DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT, DEFERRED_LIGHTING_PASS_ID_FORMAT,
        DEFERRED_PREPASS_FORMAT, copy_lighting_id::DeferredLightingIdDepthTexture,
    },
    prepass::{
        MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT, PreviousViewData,
        PreviousViewUniformOffset, PreviousViewUniforms, ViewPrepassTextures,
    },
};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, texture_2d, texture_depth_2d, uniform_buffer,
        },
        *,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewDepthStencilTexture, ViewUniform, ViewUniformOffset, ViewUniforms},
};
#[derive(Resource)]
pub(super) struct Pipelines {
    scene: BindGroupLayoutDescriptor,
    view: BindGroupLayoutDescriptor,
    copy: BindGroupLayoutDescriptor,
    surfaces: [CachedRenderPipelineId; 2],
    copy_id: CachedRenderPipelineId,
}
pub(super) fn init(
    mut commands: Commands,
    fullscreen: Res<FullscreenShader>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let scene = BindGroupLayoutDescriptor::new(
        "finite canopy prepass scene",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (26, storage_buffer_read_only_sized(false, None)),
                (27, storage_buffer_read_only_sized(false, None)),
                (28, storage_buffer_read_only_sized(false, None)),
            ),
        ),
    );
    let view = BindGroupLayoutDescriptor::new(
        "finite canopy prepass view",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<PreviousViewData>(true),
                texture_depth_2d(),
            ),
        ),
    );
    let shader = load_embedded_asset!(assets.as_ref(), "filtered_canopy_prepass.wesl");
    let surfaces = core::array::from_fn(|normal| {
        let mut targets = vec![
            Some(DEFERRED_PREPASS_FORMAT.into()),
            Some(MOTION_VECTOR_PREPASS_FORMAT.into()),
            Some(DEFERRED_LIGHTING_PASS_ID_FORMAT.into()),
        ];
        if normal == 1 {
            targets.push(Some(NORMAL_PREPASS_FORMAT.into()));
        }
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("finite canopy deferred surfaces".into()),
            layout: vec![scene.clone(), view.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: if normal == 1 {
                    vec!["WRITE_NORMAL".into()]
                } else {
                    vec![]
                },
                entry_point: Some("fragment".into()),
                targets,
                ..Default::default()
            }),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            ..Default::default()
        })
    });
    // The standard ID copy has already run in Prepass. Refresh it after this
    // custom producer so sky pixels acquiring a canopy surface have a valid ID.
    let copy = BindGroupLayoutDescriptor::new(
        "finite canopy lighting ID refresh",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Uint),
        ),
    );
    let copy_id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("finite canopy lighting ID refresh".into()),
        layout: vec![copy.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: assets
                .load("embedded://bevy_core_pipeline/deferred/copy_deferred_lighting_id.wesl"),
            ..Default::default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        ..Default::default()
    });
    commands.insert_resource(Pipelines {
        scene,
        view,
        copy,
        surfaces,
        copy_id,
    });
}
pub(super) fn render(
    view: ViewQuery<(
        &crate::prelude::SolarikLighting,
        &ViewPrepassTextures,
        &ViewDepthStencilTexture,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        &DeferredLightingIdDepthTexture,
    )>,
    gpu: Res<CanopyGpu>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    views: Res<ViewUniforms>,
    previous: Res<PreviousViewUniforms>,
    mut ctx: RenderContext,
) {
    if !gpu.active {
        return;
    }
    let (_, textures, depth, offset, previous_offset, lighting_depth) = view.into_inner();
    let (Some(gbuffer), Some(motion), Some(lighting), Some(sampled_depth)) = (
        &textures.deferred,
        &textures.motion_vectors,
        &textures.deferred_lighting_pass_id,
        &textures.depth,
    ) else {
        return;
    };
    let (Some(pipeline), Some(copy_pipeline)) = (
        cache.get_render_pipeline(pipelines.surfaces[usize::from(textures.normal.is_some())]),
        cache.get_render_pipeline(pipelines.copy_id),
    ) else {
        return;
    };
    let (Some(source), Some(regions), Some(scene), Some(view_binding), Some(previous_binding)) = (
        &gpu.sources,
        &gpu.regions,
        gpu.scene.binding(),
        views.uniforms.binding(),
        previous.uniforms.binding(),
    ) else {
        return;
    };
    let scene_group = device.create_bind_group(
        "finite canopy prepass scene",
        &cache.get_bind_group_layout(&pipelines.scene),
        &BindGroupEntries::with_indices((
            (26, source.as_entire_binding()),
            (27, regions.as_entire_binding()),
            (28, scene),
        )),
    );
    let view_group = device.create_bind_group(
        "finite canopy prepass view",
        &cache.get_bind_group_layout(&pipelines.view),
        &BindGroupEntries::sequential((
            view_binding,
            previous_binding,
            &sampled_depth.texture.default_view,
        )),
    );
    let attach = |view| {
        Some(RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        })
    };
    let mut attachments = vec![
        attach(&gbuffer.texture.default_view),
        attach(&motion.texture.default_view),
        attach(&lighting.texture.default_view),
    ];
    if let Some(normal) = &textures.normal {
        attachments.push(attach(&normal.texture.default_view));
    }
    {
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("finite canopy deferred surfaces"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth.attachment.depth_stencil_views().attachment_view(),
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &scene_group, &[]);
        pass.set_bind_group(1, &view_group, &[offset.offset, previous_offset.offset]);
        pass.draw(0..3, 0..1);
    }
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture().as_image_copy(),
        sampled_depth.texture.texture.as_image_copy(),
        textures.size,
    );
    let copy_group = device.create_bind_group(
        "finite canopy lighting ID refresh",
        &cache.get_bind_group_layout(&pipelines.copy),
        &BindGroupEntries::single(&lighting.texture.default_view),
    );
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("finite canopy lighting ID refresh"),
        color_attachments: &[],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &lighting_depth.texture.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(copy_pipeline);
    pass.set_bind_group(0, &copy_group, &[]);
    pass.draw(0..3, 0..1);
}
