//! Actual Vulkan decoding and support-point transforms for render-only placements.
#[path = "support/placement_gpu.rs"]
mod gpu;
#[path = "support/placement_reference.rs"]
mod reference;
use bevy_math::DVec3;
use bevy_solarik::scene::placement::{GpuPlacementPage, Placement12, Placement16, PlacementPage};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    root_height: [f32; 4],
    rotation: [f32; 4],
    prototype: [u32; 4],
    points: [[f32; 4]; 8],
}

#[test]
fn scalar_storage_strides_and_relative_page_contract() {
    assert_eq!(size_of::<Placement12>(), 12);
    assert_eq!(size_of::<Placement16>(), 16);
    assert_eq!(size_of::<GpuPlacementPage>(), 32);
    assert_eq!(align_of::<GpuPlacementPage>(), 16);
    assert_eq!(size_of::<Probe>(), 176);
    let page = PlacementPage {
        origin: [1e12, -1e12, 6_371_000.0],
        extent: [0.0, 25.0, 100.0],
        height_base: 1.0,
        height_span: 30.0,
    };
    let relative = page
        .relative_to([1e12 + 0.125, -1e12 - 0.25, 6_370_990.0])
        .unwrap();
    assert_eq!(relative.origin_height, [-0.125, 0.25, 10.0, 1.0]);
    assert_eq!(relative.extent_height, [0.0, 25.0, 100.0, 30.0]);
    assert!(page.relative_to([f64::NAN; 3]).is_none());
    assert!(page.relative_to([f64::MAX; 3]).is_none());
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(
            PlacementPage {
                extent: [value; 3],
                ..page
            }
            .relative_to([0.0; 3])
            .is_none()
        );
        assert!(
            PlacementPage {
                height_span: value,
                ..page
            }
            .relative_to([0.0; 3])
            .is_none()
        );
    }
    for value in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        assert!(
            PlacementPage {
                height_base: value,
                ..page
            }
            .relative_to([0.0; 3])
            .is_none()
        );
    }
}

fn check_decode(case: &reference::Case, probe: Probe, stride: usize) -> (f64, f64) {
    let (root, height, quaternion) = reference::reference(case, stride);
    assert_eq!(probe.prototype, [case.prototype, 0, 0, 0]);
    let page = case.page.relative_to(case.camera).unwrap();
    for (i, expected) in root.to_array().into_iter().chain([height]).enumerate() {
        let operands = if i < 3 {
            f64::from(page.origin_height[i].abs() + page.extent_height[i])
        } else {
            f64::from(page.origin_height[3] + page.extent_height[3])
        };
        assert!(
            (f64::from(probe.root_height[i]) - expected).abs()
                <= 4.0 * f64::from(f32::EPSILON) * (operands + 1.0),
            "root/height component {i}"
        );
    }
    for (value, expected) in probe.rotation.into_iter().zip(quaternion) {
        assert!(
            (f64::from(value) - expected).abs() <= 16.0 * f64::from(f32::EPSILON),
            "quaternion"
        );
    }
    let bound = reference::bound(case, root, height, quaternion);
    let allowance = reference::arithmetic_allowance(case, page);
    let observed = check_points(case, probe, bound + allowance);
    (observed, allowance)
}

fn check_points(case: &reference::Case, probe: Probe, limit: f64) -> f64 {
    let mut maximum: f64 = 0.0;
    let matrix = reference::matrix(case.quaternion) * case.height;
    let root = case.root - DVec3::from_array(case.camera);
    for (index, actual) in probe.points.into_iter().enumerate() {
        assert_eq!(actual[3], 1.0);
        let expected = root + matrix * reference::point(index, case.radius as f32);
        let measured = DVec3::new(
            f64::from(actual[0]),
            f64::from(actual[1]),
            f64::from(actual[2]),
        );
        let error = measured.distance(expected);
        assert!(
            error.is_finite() && error <= limit,
            "support error {error} > {limit}"
        );
        maximum = maximum.max(error);
    }
    maximum
}

#[test]
#[ignore = "requires Vulkan and SOLARIK_PLACEMENT_FIXTURE; serialize with builds/captures"]
fn packed_placement_gpu_readback() {
    futures_lite::future::block_on(async {
        let (cases, synthetic) = reference::fixture();
        let (device, queue, info) = gpu::device().await;
        let pages: Vec<_> = cases
            .iter()
            .map(|case| case.page.relative_to(case.camera).unwrap())
            .collect();
        let radii: Vec<_> = cases.iter().map(|case| case.radius as f32).collect();
        let mut summaries = Vec::new();
        for stride in [3, 4] {
            let words: Vec<_> = cases
                .iter()
                .flat_map(|case| {
                    if stride == 3 {
                        &case.words[..3]
                    } else {
                        &case.words[3..]
                    }
                })
                .copied()
                .collect();
            let probes = gpu::run(&device, &queue, &words, &pages, &radii, stride);
            summaries.push(verify(&cases, &probes, synthetic, stride));
        }
        prototype_capacity(&device, &queue, &cases[0]);
        report(&info, cases.len(), synthetic, &summaries);
    });
}

fn verify(cases: &[reference::Case], probes: &[Probe], synthetic: usize, stride: usize) -> String {
    assert_eq!(cases.len(), probes.len());
    let (mut maximum, mut far_pixel, mut allowance, mut far) = (0.0_f64, 0.0_f64, 0.0_f64, 0);
    for (index, (case, &probe)) in cases.iter().zip(probes).enumerate() {
        let (error, tolerance) = check_decode(case, probe, stride);
        maximum = maximum.max(error);
        allowance = allowance.max(tolerance);
        let distance =
            case.root.distance(DVec3::from_array(case.camera)) - case.height * case.radius - error;
        if index >= synthetic && distance >= 1536.0 {
            far_pixel =
                far_pixel.max(error * (540.0 / core::f64::consts::FRAC_PI_8.tan()) / distance);
            far += 1;
        }
    }
    assert!(far > 0 && far_pixel < 1.0);
    format!(
        "{{\"stride\":{},\"maximum_support_point_error_m\":{maximum},\"maximum_arithmetic_allowance_m\":{allowance},\"far_samples\":{far},\"maximum_far_point_error_pixels\":{far_pixel}}}",
        stride * 4
    )
}

fn prototype_capacity(device: &wgpu::Device, queue: &wgpu::Queue, case: &reference::Case) {
    let mut words = case.words[3..].to_vec();
    words[3] = (words[3] & 0xc000ffff) | (16383 << 16);
    let page = case.page.relative_to(case.camera).unwrap();
    let probe = gpu::run(device, queue, &words, &[page], &[case.radius as f32], 4);
    assert_eq!(probe[0].prototype[0], 16383);
}

#[expect(
    clippy::print_stdout,
    reason = "Recorded numerical GPU fixture receipt"
)]
fn report(info: &wgpu::AdapterInfo, count: usize, synthetic: usize, summaries: &[String]) {
    let result = format!(
        "{{\"adapter\":{:?},\"driver\":{:?},\"driver_info\":{:?},\"backend\":\"Vulkan\",\"records\":{count},\"synthetic_records\":{synthetic},\"formats\":[{}],\"full_population_gpu_accepted\":false,\"appearance_accepted\":false}}",
        info.name,
        info.driver,
        info.driver_info,
        summaries.join(","),
    );
    if let Ok(path) = std::env::var("SOLARIK_PLACEMENT_REPORT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        writeln!(file, "{result}").unwrap();
    }
    println!("{result}");
}
