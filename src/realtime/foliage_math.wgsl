#define_import_path bevy_solarik::foliage_math

// Reject a ray hit behind a raster-only occluder or on a different surface.
fn foliage_depth_matches(raster_position: vec3<f32>, hit_position: vec3<f32>, raster_normal: vec3<f32>, hit_normal: vec3<f32>, camera_distance: f32) -> bool {
    return distance(raster_position, hit_position) <= max(0.002, camera_distance * 0.00001)
        && dot(raster_normal, hit_normal) > 0.95;
}

// Compare BSDF and area-light strategies in the same solid-angle measure.
fn foliage_solid_angle_pdf(area_pdf: f32, distance_squared: f32, light_cosine: f32) -> f32 {
    return area_pdf * distance_squared / max(abs(light_cosine), 0.000001);
}

