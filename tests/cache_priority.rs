//! Actual production scheduling, indirect blending and readbacks on Vulkan.
const CAPACITY: usize = 1 << 20;
const BUDGET: usize = 40_000;
const SENTINEL: u32 = 0xdead_beef;
const CONFIG: usize = 0;
const SELECTED: usize = 1;
const METRICS: usize = 2;
const GEOMETRY: usize = 3;
const RADIANCE: usize = 4;
const ACTIVE: usize = 5;
const COUNT: usize = 6;
const NEW_RADIANCE: usize = 7;
const DELTAS: usize = 8;
const DISPATCH: usize = 9;

fn item<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(name).expect("production item");
    let body = start + source[start..].find('{').expect("body");
    let mut depth = 0;
    for (offset, ch) in source[body..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=body + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated item");
}

fn shader() -> String {
    let priority = include_str!("../src/realtime/world_cache_priority.wgsl");
    let update = include_str!("../src/realtime/world_cache_update.wgsl");
    let bindings = include_str!("../src/realtime/realtime_bindings.wgsl");
    format!(
        r#"
const WORLD_CACHE_CELL_UPDATES_SOFT_CAP = 40000u;
const WORLD_CACHE_MAX_TEMPORAL_SAMPLES = 32.0;
{}
@group(0) @binding(0) var<uniform> constants:PushConstants;
@group(0) @binding(1) var<storage,read_write> world_cache_a:array<u32>;
@group(0) @binding(2) var<storage,read_write> world_cache_b:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> world_cache_geometry_data:array<WorldCacheGeometryData>;
@group(0) @binding(4) var<storage,read_write> world_cache_radiance:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> world_cache_active_cell_indices:array<u32>;
@group(0) @binding(6) var<storage,read_write> world_cache_active_cells_count:u32;
@group(0) @binding(7) var<storage,read_write> world_cache_active_cells_new_radiance:array<vec3<f32>>;
@group(0) @binding(8) var<storage,read_write> world_cache_luminance_deltas:array<f32>;
fn luminance(v:vec3<f32>)->f32 {{ return dot(v,vec3(0.2126,0.7152,0.0722)); }}
{}
{}
{}
@compute @workgroup_size(64)
{}
"#,
        item(bindings, "struct PushConstants"),
        item(bindings, "struct WorldCacheGeometryData"),
        &priority[priority
            .find("const PRIORITY_BUCKET_COUNT")
            .expect("scheduler")..],
        item(update, "fn cache_blend_amount("),
        item(update, "fn blend_new_samples(")
    )
}

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    empty_group: wgpu::BindGroup,
    dispatch_group: wgpu::BindGroup,
    readback: wgpu::Buffer,
}

fn buffer(device: &wgpu::Device, bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn layout_entry(binding: u32, uniform: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only: false }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Fixture {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan");
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            })
            .await
            .expect("device");
        Self::create(device, queue)
    }

    fn create(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let sizes = [
            16,
            CAPACITY * 4,
            4096,
            CAPACITY * 32,
            CAPACITY * 16,
            CAPACITY * 4,
            4,
            CAPACITY * 16,
            CAPACITY * 4,
            12,
        ];
        let buffers = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let usage = if i == CONFIG {
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                } else if i == DISPATCH {
                    storage | wgpu::BufferUsages::INDIRECT
                } else {
                    storage
                };
                buffer(&device, size, usage)
            })
            .collect::<Vec<_>>();
        let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &(0..9).map(|i| layout_entry(i, i == 0)).collect::<Vec<_>>(),
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
        let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[layout_entry(0, false)],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &buffers[..9]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &empty,
            entries: &[],
        });
        let dispatch_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &dispatch_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers[DISPATCH].as_entire_binding(),
            }],
        });
        let pipelines = Self::pipelines(&device, &group_layout, &empty, &dispatch_layout);
        let readback = buffer(
            &device,
            (BUDGET + 64) * 4 + 4096 + CAPACITY * 48,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        Self {
            device,
            queue,
            buffers,
            pipelines,
            group,
            empty_group,
            dispatch_group,
            readback,
        }
    }

    fn pipelines(
        device: &wgpu::Device,
        group: &wgpu::BindGroupLayout,
        empty: &wgpu::BindGroupLayout,
        dispatch: &wgpu::BindGroupLayout,
    ) -> Vec<wgpu::ComputePipeline> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production radiance cache priority and blending"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        [
            "clear_world_cache_priority",
            "histogram_world_cache_priority",
            "budget_world_cache_priority",
            "select_world_cache_priority",
            "blend_new_samples",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, entry)| {
            let groups = if i == 4 {
                vec![group, empty]
            } else {
                vec![group, empty, dispatch]
            };
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &groups.iter().map(|g| Some(*g)).collect::<Vec<_>>(),
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect()
    }

    fn initialize(&self, unlit: usize, old: usize, last_frame: u32) {
        let count = unlit + old;
        let geometry = (0..count)
            .map(|_| [0, 0, 0, last_frame, 0, 0, 0, 0])
            .collect::<Vec<_>>();
        let radiance: Vec<[f32; 4]> = (0..count)
            .map(|i| {
                if i < unlit {
                    [0.0; 4]
                } else {
                    [10.0, 10.0, 10.0, 1.0]
                }
            })
            .collect::<Vec<_>>();
        if count != 0 {
            self.queue
                .write_buffer(&self.buffers[GEOMETRY], 0, bytemuck::cast_slice(&geometry));
            self.queue
                .write_buffer(&self.buffers[RADIANCE], 0, bytemuck::cast_slice(&radiance));
            self.queue.write_buffer(
                &self.buffers[ACTIVE],
                0,
                bytemuck::cast_slice(&(0..count as u32).collect::<Vec<_>>()),
            );
        }
        self.queue
            .write_buffer(&self.buffers[COUNT], 0, bytemuck::bytes_of(&(count as u32)));
        self.queue.write_buffer(
            &self.buffers[NEW_RADIANCE],
            0,
            bytemuck::cast_slice(&vec![[10.0f32; 4]; BUDGET]),
        );
        self.queue.write_buffer(
            &self.buffers[DELTAS],
            0,
            bytemuck::cast_slice(&vec![0.0f32; count.max(1)]),
        );
    }

    fn frame(&self, frame: u32, count: usize) -> Report {
        self.queue.write_buffer(
            &self.buffers[CONFIG],
            0,
            // Exercise the production distinction: stochastic seed != frame age.
            bytemuck::cast_slice(&[frame.wrapping_mul(5_782_582), 0, frame, 0]),
        );
        self.queue.write_buffer(
            &self.buffers[SELECTED],
            0,
            bytemuck::cast_slice(&vec![SENTINEL; BUDGET + 64]),
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.group, &[]);
            pass.set_bind_group(1, &self.empty_group, &[]);
            pass.set_bind_group(2, &self.dispatch_group, &[]);
            for (i, groups) in [1, (CAPACITY / 256) as u32, 1, (CAPACITY / 256) as u32]
                .into_iter()
                .enumerate()
            {
                pass.set_pipeline(&self.pipelines[i]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            pass.set_bind_group(2, None, &[]);
            pass.set_pipeline(&self.pipelines[4]);
            pass.dispatch_workgroups_indirect(&self.buffers[DISPATCH], 0);
        }
        self.copy_results(&mut encoder);
        self.queue.submit([encoder.finish()]);
        self.read(frame, count)
    }

    fn copy_results(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut offset = 0;
        for (index, size) in [
            (SELECTED, (BUDGET + 64) * 4),
            (METRICS, 4096),
            (GEOMETRY, CAPACITY * 32),
            (RADIANCE, CAPACITY * 16),
        ] {
            encoder.copy_buffer_to_buffer(
                &self.buffers[index],
                0,
                &self.readback,
                offset,
                size as u64,
            );
            offset += size as u64;
        }
    }

    fn read(&self, frame: u32, count: usize) -> Report {
        let (sender, receiver) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                sender.send(r).expect("callback");
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU");
        receiver.recv().expect("callback").expect("map");
        let data = self.readback.slice(..).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let metrics = words[BUDGET + 64..BUDGET + 64 + 1024].to_vec();
        let selected = metrics[96] as usize;
        assert_eq!(
            selected,
            count.min(BUDGET),
            "hard budget and work conservation"
        );
        assert_eq!(
            metrics[101] as usize, selected,
            "actual production indirect blend count"
        );
        assert_eq!(
            words[selected], SENTINEL,
            "partial workgroups may not write a tail"
        );
        let mut keys = words[..selected].to_vec();
        keys.sort_unstable();
        assert!(
            keys.windows(2).all(|p| p[0] != p[1]),
            "each selected cell occurs once"
        );
        let geometry_start = BUDGET + 64 + 1024;
        let radiance_start = geometry_start + CAPACITY * 8;
        for &key in &keys {
            assert!((key as usize) < count, "selected cell must be active");
            assert_eq!(words[geometry_start + key as usize * 8 + 3], frame);
            assert_eq!(
                &words[radiance_start + key as usize * 4..radiance_start + key as usize * 4 + 3],
                &[10.0f32.to_bits(); 3]
            );
        }
        let last_traced = (0..count)
            .map(|i| words[geometry_start + i * 8 + 3])
            .collect();
        let unlit_remaining = (0..count)
            .filter(|&i| words[radiance_start + i * 4 + 3] == 0)
            .count();
        drop(data);
        self.readback.unmap();
        Report {
            metrics,
            keys,
            last_traced,
            unlit_remaining,
        }
    }
}

struct Report {
    metrics: Vec<u32>,
    keys: Vec<u32>,
    last_traced: Vec<u32>,
    unlit_remaining: usize,
}

#[test]
#[ignore = "requires Vulkan; serialize with Cargo builds and native captures"]
fn production_priority_budget_and_recovery_gpu() {
    futures_lite::future::block_on(async {
        let f = Fixture::new().await;
        for (unlit, old) in [
            (0, 0),
            (7, 0),
            (70_003, 0),
            (0, 70_003),
            (40_000, 60_003),
            (CAPACITY, 0),
        ] {
            f.initialize(unlit, old, 0);
            let r = f.frame(8, unlit + old);
            let expected_new = if old == 0 {
                unlit.min(BUDGET)
            } else {
                unlit.min(BUDGET - old.min(BUDGET / 5))
            };
            assert_eq!(r.metrics[97] as usize, unlit);
            assert_eq!(r.metrics[98] as usize, expected_new);
            assert_eq!(
                r.keys.iter().filter(|&&k| (k as usize) < unlit).count(),
                expected_new,
                "unlit-first priority retains a refresh reserve"
            );
        }
        f.initialize(100_000, 0, 0);
        for frame in 1..=3 {
            let r = f.frame(frame, 100_000);
            if frame == 3 {
                assert_eq!(r.unlit_remaining, 0);
            }
        }
        f.initialize(0, 100_000, 0);
        for frame in 1..=4 {
            let r = f.frame(frame, 100_000);
            if frame == 4 {
                assert!(r.last_traced.iter().all(|&t| t > 0));
            }
        }
        f.initialize(0, 80_000, u32::MAX - 2);
        let young = vec![[0u32, 0, 0, 4, 0, 0, 0, 0]; 40_000];
        f.queue.write_buffer(
            &f.buffers[GEOMETRY],
            40_000 * 32,
            bytemuck::cast_slice(&young),
        );
        let r = f.frame(5, 80_000);
        assert_eq!(r.metrics[100], 8, "age subtraction survives frame wrap");
        assert!(
            r.keys.iter().all(|&k| k < 40_000),
            "old cells precede newly refreshed cells"
        );
    });
}
