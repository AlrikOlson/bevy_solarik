//! A persistent GPU consumer must see the logical scene, including within capacity.
#![expect(
    clippy::print_stdout,
    reason = "The explicit GPU regression emits adapter and readback evidence under --nocapture."
)]
use super::*;
use bevy_render::renderer::WgpuWrapper;
use core::time::Duration;

fn rows(count: u32, frame: u32) -> Vec<[u32; 4]> {
    let mut items: Vec<_> = (0..count)
        .map(|i| {
            let transform = Affine3A::from_translation(Vec3::X * i as f32 * 8.);
            let instance = Instance {
                id: Default::default(),
                transform,
                previous: transform,
                identity: u64::from(i),
                rays: true,
            };
            let bounds = transformed_bounds(-Vec3::ONE, Vec3::ONE, transform).unwrap();
            (instance, i * 256, bounds)
        })
        .collect();
    pack(&mut items, frame)
}

async fn device() -> (RenderDevice, RenderQueue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    println!("CANOPY_BINDING_ADAPTER {:?}", adapter.get_info());
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    (
        RenderDevice::from(device),
        RenderQueue(Arc::new(WgpuWrapper::new(queue))),
    )
}

struct Probe {
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    group: Option<wgpu::BindGroup>,
}

impl Probe {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bounded canopy binding probe"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("filtered_canopy_binding_probe.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bounded canopy binding probe"),
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("canopy binding probe result"),
                size: 16,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            pipeline,
            output: buffer(BufferUsages::STORAGE | BufferUsages::COPY_SRC),
            readback: buffer(BufferUsages::COPY_DST | BufferUsages::MAP_READ),
            group: None,
        }
    }

    fn bind(&mut self, device: &wgpu::Device, gpu: &CanopyGpu) {
        if self.group.is_some() && !gpu.bindings_changed {
            return;
        }
        self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("persistent canopy consumer"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.scene.binding().unwrap(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output.as_entire_binding(),
                },
            ],
        }));
    }

    fn read(&self, device: &wgpu::Device, queue: &RenderQueue) -> [u32; 4] {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, 16);
        queue.submit([encoder.finish()]);
        self.readback
            .map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .unwrap();
        let result =
            bytemuck::from_bytes::<[u32; 4]>(&self.readback.get_mapped_range(..)).to_owned();
        self.readback.unmap();
        result
    }
}

#[test]
#[ignore = "Serialized Vulkan persistent canopy binding regression"]
fn canopy_scene_growth_and_shrink_rebind_within_retained_allocation() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let mut gpu = CanopyGpu::default();
        publish_scene(&mut gpu, rows(16, 0), &device, &queue);
        let allocation = gpu.scene.buffer().unwrap().id();
        let mut probe = Probe::new(device.wgpu_device());
        let mut previous_count = None;
        for (frame, count) in [0, 1, 2, 3, 2, 0, 0].into_iter().enumerate() {
            gpu.bindings_changed = false;
            let rows = rows(count, frame as u32 + 1);
            let expected = [rows.len() as u32, rows[0][0], count, 0];
            publish_scene(&mut gpu, rows, &device, &queue);
            assert_eq!(gpu.scene.buffer().unwrap().id(), allocation);
            probe.bind(device.wgpu_device(), &gpu);
            let actual = probe.read(device.wgpu_device(), &queue);
            println!(
                "CANOPY_BINDING count={count} rebound={} actual={actual:?} expected={expected:?}",
                gpu.bindings_changed
            );
            assert_eq!(
                actual, expected,
                "consumer bound range must contain every scene node and instance"
            );
            if previous_count == Some(count) {
                assert!(
                    !gpu.bindings_changed,
                    "frame-only updates reuse the binding"
                );
            }
            previous_count = Some(count);
        }
    });
}
