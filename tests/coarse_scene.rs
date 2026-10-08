//! Actual production upload/lookup/decoder over the immutable measured sources.
#[path = "support/coarse_scene_gpu.rs"]
mod gpu;
#[path = "support/coarse_scene_requests.rs"]
mod requests;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_cache,
    coarse_scene::{MISSING, PackedSource},
};
use std::{io::Write, path::Path, time::Instant};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Request {
    point: [f32; 4],
    parameters: [u32; 4],
}
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    address: [u32; 4],
    measurements: coarse_cache::Directional,
}

#[test]
#[ignore = "serialized actual Vulkan production coarse cache consumer"]
fn immutable_coarse_sources_decode_and_lookup_exactly_on_gpu() {
    let source = std::env::var("SOLARIK_COARSE_INPUT").expect("cache directory");
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").expect("fresh output directory");
    let key = std::env::var("SOLARIK_COARSE_IDENTITY").expect("cache identity");
    assert_eq!(key.len(), 64);
    let identity =
        core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap());
    assert_eq!(size_of::<Probe>(), 128);
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = gpu::pipeline(device, include_str!("coarse_scene.wgsl"));
        let mut reports = Vec::new();
        for id in [5, 43] {
            for resolution in [16, 32] {
                let started = Instant::now();
                let path = Path::new(&source).join(format!("source-{id}-r{resolution}.sfcg"));
                assert!(path.metadata().unwrap().len() <= coarse_cache::MAX_CACHE_BYTES as u64);
                let bytes = std::fs::read(path).unwrap();
                let original = coarse_cache::decode(&bytes, identity).unwrap();
                assert_eq!((original.prototype, original.resolution), (id, resolution));
                let packed = PackedSource::decode(&bytes, identity).unwrap();
                let (requests, expected) = requests::build(&original, &packed);
                let (rows, total_gpu_bytes) =
                    gpu::run(&resources.0, queue, &pipeline, &packed, &requests);
                assert_eq!(rows.len(), expected.len());
                let mut absent = 0usize;
                let mut sampled_empty = 0usize;
                for (index, (actual, expected)) in rows.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual.address, expected.address,
                        "source{id} r{resolution} request{index}"
                    );
                    assert_eq!(
                        bytemuck::bytes_of(&actual.measurements),
                        bytemuck::bytes_of(&expected.measurements),
                        "source{id} r{resolution} request{index}"
                    );
                    if actual.address[0] == 0 {
                        absent += 1;
                    } else if actual.measurements.counts[1] == 0 {
                        sampled_empty += 1;
                    }
                }
                assert!(absent > 0 && sampled_empty > 0);
                assert!(packed.bytes() < bytes.len());
                reports.push(format!(
                "{{\"prototype\":{id},\"resolution\":{resolution},\"cells\":{},\"queries\":{},\"missing_queries\":{absent},\"zero_coverage_queries\":{sampled_empty},\"cache_bytes\":{},\"runtime_gpu_bytes\":{},\"all_fixture_gpu_bytes\":{total_gpu_bytes},\"seconds\":{}}}",
                original.cells.len(),rows.len(),bytes.len(),packed.bytes(),started.elapsed().as_secs_f64()));
            }
        }
        let report = format!(
            "{{\"adapter\":{:?},\"sources\":[{}],\"bit_exact\":true}}\n",
            format!("{:?}", **resources.2),
            reports.join(",")
        );
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(Path::new(&output).join("gpu.json"))
            .unwrap();
        file.write_all(report.as_bytes()).unwrap();
    });
}
