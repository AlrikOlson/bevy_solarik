use super::input::Input;
use wgpu::util::DeviceExt;

pub struct Scene {
    pub tlas: wgpu::Tlas,
    pub buffers: [wgpu::Buffer; 5],
    pub alpha: wgpu::Texture,
    pub sampler: wgpu::Sampler,
    pub explicit_bytes: u64,
}

pub fn build(device: &wgpu::Device, queue: &wgpu::Queue, input: &Input) -> Scene {
    let data: [&[u8]; 5] = [
        bytemuck::cast_slice(&input.positions),
        bytemuck::cast_slice(&input.uvs),
        bytemuck::cast_slice(&input.indices),
        bytemuck::cast_slice(&input.parts),
        bytemuck::cast_slice(&input.rays),
    ];
    let buffers = data.map(|bytes| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("source coverage input"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST,
        })
    });
    let tlas = acceleration(device, queue, input, &buffers);
    let alpha = texture(device, queue, input);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        min_filter: wgpu::FilterMode::Linear,
        mag_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let explicit_bytes =
        data.iter().map(|d| d.len() as u64).sum::<u64>() + input.alpha.len() as u64;
    assert!(explicit_bytes + input.rays.len() as u64 * 128 < 1 << 30);
    Scene {
        tlas,
        buffers,
        alpha,
        sampler,
        explicit_bytes,
    }
}

fn acceleration(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &Input,
    buffers: &[wgpu::Buffer; 5],
) -> wgpu::Tlas {
    let sizes: Vec<_> = input
        .parts
        .iter()
        .map(|part| wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: input.positions.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(part[1]),
            flags: wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION,
        })
        .collect();
    let blases: Vec<_> = sizes
        .iter()
        .map(|size| {
            device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("source reference BLAS"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            )
        })
        .collect();
    let builds: Vec<_> = blases
        .iter()
        .zip(&sizes)
        .zip(&input.parts)
        .map(|((blas, size), part)| wgpu::BlasBuildEntry {
            blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: &buffers[0],
                first_vertex: 0,
                vertex_stride: 16,
                index_buffer: Some(&buffers[2]),
                first_index: Some(part[0]),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("source reference TLAS"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: blases.len() as u32,
    });
    for (i, blas) in blases.iter().enumerate() {
        *tlas.get_mut_single(i).unwrap() = Some(wgpu::TlasInstance::new(
            blas,
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            i as u32,
            255,
        ));
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(&builds, [&tlas]);
    queue.submit([encoder.finish()]);
    tlas
}

fn texture(device: &wgpu::Device, queue: &wgpu::Queue, input: &Input) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("exact level-zero alpha"),
        size: wgpu::Extent3d {
            width: input.width,
            height: input.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &input.alpha,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(input.width * 4),
            rows_per_image: Some(input.height),
        },
        texture.size(),
    );
    texture
}
