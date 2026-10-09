// Reuse verified upload, material probe and dispatch without changing its producer key.
include!("coarse_appearance_gpu.rs");
pub fn spatial_pipeline(device: &wgpu::Device, entry: &str) -> wgpu::ComputePipeline {
    let shared = [
        bevy_solarik::coarse_spatial::PACK_SHADER,
        bevy_solarik::coarse_appearance::SHADER,
    ]
    .map(|s| {
        s.lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .join("\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("source spatial materials"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "enable wgpu_ray_query;\n{shared}\n{}",
                include_str!("../coarse_spatial_material.wgsl")
            )
            .into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
