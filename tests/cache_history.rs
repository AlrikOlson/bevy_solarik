//! Production cache blending bounds expected decay in frames under update throttling.
use wgpu::util::DeviceExt;

fn function(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("fn {name}("))
        .expect("production function");
    let body = start + source[start..].find('{').expect("body");
    let mut depth = 0;
    for (offset, ch) in source[body..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=body + offset].to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function");
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn cache_history_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let gi = include_str!("../src/realtime/world_cache_update.wgsl");

        let mut source = r#"
@group(0) @binding(0) var<storage,read> config:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
const WORLD_CACHE_MAX_TEMPORAL_SAMPLES=32.0;
const WORLD_CACHE_EMPTY_CELL=0u;
var<private> world_cache_life:array<u32,1>;
var<private> scene_history_regions:array<vec4f,16>;
var<private> world_cache_checksums:array<u32,1>;
var<private> world_cache_radiance:array<vec4<f32>,1>;
var<private> world_cache_luminance_deltas:array<f32,1>;
@compute @workgroup_size(1)
fn probe() {
 let p=config[0].x;
 let alpha=cache_blend_amount(32.0,10.0,0.0,p);
 var expected=1.0;
 for(var frame=0u;frame<64u;frame++) { expected*=1.0-p*alpha; }
 output[0]=vec4(alpha,expected,mix(10.0,10.0,alpha),0.0);
 for(var reset=0u;reset<2u;reset++) {
  world_cache_life[0]=10u;
  world_cache_checksums[0]=10u;
  world_cache_radiance[0]=vec4(10.0);
  world_cache_luminance_deltas[0]=10.0;
  decay_world_cache_cell(0u,reset==1u);
  output[reset+1u]=vec4(f32(world_cache_life[0]),f32(world_cache_checksums[0]),world_cache_radiance[0].x,world_cache_luminance_deltas[0]);
 }
 output[3]=vec4(
  f32(cache_support_is_valid(vec3(0.0),10.0,1000.0)),
  f32(cache_support_is_valid(vec3(0.0),2000.0,1000.0)),
  f32(cache_support_is_valid(vec3(900.0,0.0,0.0),100.0,1000.0)),
  f32(cache_support_is_valid(vec3(0.0),bitcast<f32>(0x7f800000u),1000.0)));
 scene_history_regions[0]=vec4f(100.,-2.,-2.,1.);
 scene_history_regions[1]=vec4f(102.,2.,2.,1.);
 output[5]=vec4f(f32(scene_support_is_valid(vec3f(0.),50.)),
  f32(scene_support_is_valid(vec3f(99.,0.,0.),1.)),
  f32(scene_support_is_valid(vec3f(101.,0.,0.),0.)),
  f32(scene_support_is_valid(vec3f(0.),bitcast<f32>(0x7fc00000u))));
 output[6]=vec4f(f32(scene_support_is_valid(vec3f(bitcast<f32>(0x7fc00000u),0.,0.),1.)),
  f32(scene_support_is_valid(vec3f(0.),-1.)),0.,0.);
 output[4]=vec4(
  f32(cache_support_is_valid(vec3(0.0),-1.0,1000.0)),
  f32(cache_support_is_valid(vec3(0.0),bitcast<f32>(0x7fc00000u),1000.0)),
  f32(cache_support_is_valid(vec3(0.0),0.0,0.0)),
  f32(cache_support_is_valid(vec3(0.0),0.0,1.0)));
}
"#
        .to_owned();
        source.push_str(&function(gi, "cache_blend_amount"));
        source.push_str(&function(
            include_str!("../src/realtime/world_cache_compact.wgsl"),
            "decay_world_cache_cell",
        ));
        source.push_str(&function(
            include_str!("../src/realtime/world_cache_compact.wgsl"),
            "cache_support_is_valid",
        ));
        source.push_str(&function(
            include_str!("../src/scene/raytracing_scene_bindings.wgsl"),
            "scene_support_is_valid",
        ));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production cache temporal response"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 7 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for probability in [1.0f32, 0.5, 0.25, 0.125] {
            let tint = [probability, 0.0, 0.0, 0.0];
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[tint]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                sender.send(r).expect("callback");
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU");
            receiver.recv().expect("callback").expect("map");
            let data = readback.slice(..).get_mapped_range();
            let samples: &[[f32; 4]] = bytemuck::cast_slice(&data);
            assert!(samples[0][0] > 0.0 && samples[0][0] <= 1.0);
            assert!(
                samples[0][1] < 0.02,
                "64-frame residual at p={probability}: {:?}",
                samples[0]
            );
            assert_eq!(samples[0][2], 10.0, "constant light must retain its energy");
            assert_eq!(
                samples[1],
                [9.0, 10.0, 10.0, 10.0],
                "normal decay preserves live radiance"
            );
            assert_eq!(samples[2], [0.0; 4], "reset must discard all cache history");
            assert_eq!(
                samples[3],
                [1.0, 0.0, 0.0, 0.0],
                "bounded, long and boundary support"
            );
            assert_eq!(
                samples[4],
                [0.0, 0.0, 0.0, 1.0],
                "unknown support fails closed"
            );
            assert_eq!(
                samples[5],
                [1., 0., 0., 0.],
                "regional support preserves remote cells and rejects boundary/intersection/unknown"
            );
            assert_eq!(
                samples[6], [0.; 4],
                "nonfinite positions and negative support cannot be reused"
            );
            drop(data);
            readback.unmap();
        }
    });
}
