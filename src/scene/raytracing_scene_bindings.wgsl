enable wgpu_ray_query;

#define_import_path bevy_solarik::scene_bindings
#import bevy_solarik::scene_hit::{SceneHit, scene_miss, scene_triangle, scene_hit_is_triangle, SCENE_HIT_INVALID}

#import bevy_solarik::collimated::collimated_weight
#import bevy_solarik::ring_transport::ring_shadow
#import bevy_solarik::detail_sampling::{DetailCoordinates, sample_surface_detail, detail_normal}
#import bevy_solarik::light_medium::{LightMedium, medium_transmittance, cloud_transmission, MEDIUM_CLOUD_COLUMN}
#import bevy_pbr::lighting::perceptualRoughnessToRoughness
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace

struct InstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    light_probability: f32,
}

struct VertexBuffer { vertices: array<PackedVertex> }

struct IndexBuffer { indices: array<u32> }

struct PackedVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

fn unpack_vertex(packed: PackedVertex) -> Vertex {
    var vertex: Vertex;
    vertex.position = packed.a.xyz;
    vertex.normal = vec3(packed.a.w, packed.b.xy);
    vertex.uv = packed.b.zw;
    vertex.tangent = packed.tangent;
    return vertex;
}

struct Material {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: vec3<f32>,
    perceptual_roughness: f32,
    emissive: vec3<f32>,
    metallic: f32,
    // Alpha handling for the rays (upstream's padding): the mask cutoff, the
    // MATERIAL_FLAG_* bits and the base colour's alpha factor.
    alpha_cutoff: f32,
    flags: u32,
    base_color_alpha: f32,
    reflectance: f32,
    emission_cone: vec4<f32>,
    gaussian: vec4<f32>,
}

// Mirrored in binder.rs. Exactly one of OPAQUE / ALPHA_MASK / ALPHA_BLEND.
const MATERIAL_FLAG_OPAQUE = 1u;
const MATERIAL_FLAG_ALPHA_MASK = 2u;
const MATERIAL_FLAG_ALPHA_BLEND = 4u;
const MATERIAL_FLAG_DIFFUSE_BLEND = 16u;
const MATERIAL_FLAG_DOUBLE_SIDED = 8u;

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

const MIRROR_ROUGHNESS_THRESHOLD = 0.001f;

struct LightSource {
    // Low bit clear: an emissive mesh, its triangle count in the bits above
    // (upstream). Low bit set: another kind of light, selected by the bits
    // above. Mirrored in binder.rs.
    kind: u32,
    id: u32,
    selection_probability: f32,
    alias_threshold: f32,
    alias_index: u32,
}

const LIGHT_SOURCE_KIND_EMISSIVE_MESH = 0u;
const LIGHT_SOURCE_KIND_DIRECTIONAL = 1u;
const LIGHT_SOURCE_KIND_POINT = 3u;
const LIGHT_SOURCE_KIND_SPOT = 5u;

fn light_source_is_emissive_mesh(light_source: LightSource) -> bool {
    return (light_source.kind & 1u) == 0u;
}

// A point or spot light as a sphere light of the light's radius, in the
// raster path's units (bevy_pbr's candela). Mirrors GpuLocalLight in
// binder.rs.
struct LocalLight {
    position: vec3<f32>,
    radius: f32,
    // Radiance of the sphere's surface, intensity / (π r²).
    radiance: vec3<f32>,
    // The sphere's area, the inverse pdf of a uniform area sample.
    inverse_pdf: f32,
    // Spot cone axis (unused for point lights).
    direction: vec3<f32>,
    // The raster path's cut-off (its smooth window applies).
    range: f32,
    // cos of the outer and inner cone angles; -1 for both means no cone.
    cos_outer: f32,
    cos_inner: f32,
    _padding: vec2<f32>,
}

// The sky: a cubemap in the scene's radiance units, scaled by `intensity`
// (0 when the scene has no sky).
struct SkyLight {
    intensity: f32,
    ray_max_distance: f32,
    relative_ray_min: f32,
    material_transport_flags: u32,
    // The air of the planet the scene stands on, which directional light
    // crosses on its way to a surface. No medium when its radius is zero.
    medium: LightMedium,
}

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

const LIGHT_NOT_PRESENT_THIS_FRAME = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage> vertex_buffers: binding_array<VertexBuffer>;
@group(0) @binding(1) var<storage> index_buffers: binding_array<IndexBuffer>;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var samplers: binding_array<sampler>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var tlas: acceleration_structure;
@group(0) @binding(6) var<storage> transforms: array<mat4x4<f32>>; // TODO: Use mat3x4<f32>?
@group(0) @binding(7) var<storage> previous_frame_transforms: array<mat4x4<f32>>; // TODO: Use mat3x4<f32>?
@group(0) @binding(8) var<storage> geometry_ids: array<InstanceGeometryIds>;
@group(0) @binding(9) var<storage> material_ids: array<u32>; // TODO: Store material_id in instance_custom_data instead?
@group(0) @binding(10) var<storage> light_sources: array<LightSource>;
@group(0) @binding(11) var<storage> directional_lights: array<DirectionalLight>;
@group(0) @binding(12) var<storage> local_lights: array<LocalLight>;
@group(0) @binding(13) var<storage> previous_frame_light_id_translations: array<u32>;
@group(0) @binding(14) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(15) var brdf_dfg_lut_sampler: sampler;
@group(0) @binding(16) var sky_texture: texture_cube<f32>;
@group(0) @binding(17) var sky_sampler: sampler;
@group(0) @binding(18) var<storage> sky_light: SkyLight; // storage: uniforms can't share a group with binding arrays
// The planet's weather cube, as the atmosphere pass reads it: R cover, G
// cloud-top height as a fraction of the cloud layer, B extinction scale.
@group(0) @binding(19) var weather_texture: texture_cube<f32>;
@group(0) @binding(20) var weather_sampler: sampler;

struct SurfaceDetailParameters {
    coordinates: DetailCoordinates,
    textures: vec4u,
    // Mesoscale colour (x) and detail (y) atlas indices.
    meso: vec4u,
}
@group(0) @binding(23) var<storage> surface_details: array<SurfaceDetailParameters>;
// Eight old/new unions, two vec4s each. A zero w denotes an empty octant.
@group(0) @binding(25) var<storage> scene_history_regions: array<vec4f>;
fn scene_support_is_valid(position: vec3f, radius: f32) -> bool {
    if !(radius >= 0.0 && radius < 3.4e38) || !all(abs(position) < vec3f(3.4e38)) { return false; }
    for (var region = 0u; region < 8u; region++) {
        let lower = scene_history_regions[region*2u];
        if lower.w == 0.0 { continue; }
        let upper = scene_history_regions[region*2u+1u];
        let delta = max(max(lower.xyz-position, position-upper.xyz), vec3f(0.0));
        if !(length(delta) > radius) { return false; }
    }
    return true;
}

// Radiance arriving from the sky along `direction` (world space, pointing
// away from the surface), for a ray that left the scene. Black when the
// scene has no sky. Cube maps are left-handed, so z is negated the way
// Bevy's skybox and environment map shaders do it.
fn sample_sky(direction: vec3<f32>) -> vec3<f32> {
    let cube_direction = vec3(direction.xy, -direction.z);
    return textureSampleLevel(sky_texture, sky_sampler, cube_direction, 0.0).rgb * sky_light.intensity;
}

// Share of a directional light that reaches `origin` from `direction`
// through the planet's air: a low sun is dimmer and redder at the ground
// than a high one. One when the scene has no planetary atmosphere.
fn directional_light_transmittance(origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return medium_transmittance(sky_light.medium, origin, direction) * cloud_shadow(origin, direction) * ring_shadow(origin, direction);
}

// Distance along `direction` at which a ray from `p`, inside a sphere of
// `radius` about the origin, leaves it.
fn sphere_exit(p: vec3<f32>, direction: vec3<f32>, radius: f32) -> f32 {
    let b = dot(p, direction);
    return -b + sqrt(max(b * b - dot(p, p) + radius * radius, 0.0));
}

// Share of a directional light that the planet's cloud lets through to
// `origin`: the cloud is looked up where the ray to the light crosses its
// base and the middle of its depth, so a shadow lies down-sun of its cloud
// and further away the lower the sun. Partial cover shades that share of
// the light. The world's axes are taken to be the planet's body-fixed axes.
fn cloud_shadow(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let m = sky_light.medium;
    if m.radius <= 0.0 || m.cloud.w <= 0.0 { return 1.0; }
    let p = origin - m.centre;
    let r = length(p);
    if r >= m.cloud.y { return 1.0; }
    let at_base = p + direction * select(0.0, sphere_exit(p, direction, m.cloud.x), r < m.cloud.x);
    let mu = dot(normalize(at_base), direction);
    // Below the horizon the planet itself is in the way.
    if mu <= 0.0 { return 1.0; }
    let low = textureSampleLevel(weather_texture, weather_sampler, normalize(at_base), 0.0);
    let shell = m.cloud.y - m.cloud.x;
    let middle = m.cloud.x + 0.5 * low.g * shell;
    let at_middle = p + direction * select(0.0, sphere_exit(p, direction, middle), r < middle);
    let high = textureSampleLevel(weather_texture, weather_sampler, normalize(at_middle), 0.0);
    let weather = 0.5 * (low + high);
    if weather.r <= 0.0 { return 1.0; }
    let thickness = weather.g * shell;
    // Inside the layer only the cloud above the point shades it.
    let above = clamp((m.cloud.x + thickness - r) / max(thickness, 1.0), 0.0, 1.0);
    let column = m.cloud.z * weather.b * thickness * MEDIUM_CLOUD_COLUMN * above;
    return 1.0 - weather.r * (1.0 - cloud_transmission(mu, column));
}

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

// Raster positions are reconstructed with f32 matrix products/division. A
// fixed millimetre vanishes at planetary magnitudes. Eight f32 epsilons provide an allowance for
// local reconstruction arithmetic with rigid camera-relative transforms;
// this is a conservative numerical policy, not a general sheared-instance bound.
// Only the visibility origin moves; the BRDF and physical surface do not.
fn offset_surface_ray(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let magnitude = max(max(abs(position.x), abs(position.y)), abs(position.z));
    let distance = max(RAY_T_MIN, magnitude * 0.00000095367431640625);
    return position + normal * distance;
}

// RAY_T_MAX retains the historical default for external shader compatibility.
fn ray_max_distance() -> f32 { return sky_light.ray_max_distance; }

const RAY_NO_CULL = 0xFFu;

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> SceneHit {
    return trace_ray_impl(ray_origin, ray_direction, ray_t_min, ray_t_max, ray_flag, false);
}

// Camera and pathtracer bounce rays can interact with glass. Shadow and
// realtime GI callers retain trace_ray's transparent-to-light behavior.
fn trace_glass_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32) -> SceneHit {
    return trace_ray_impl(ray_origin, ray_direction, ray_t_min, ray_t_max, RAY_FLAG_NONE, true);
}

fn scene_hit_from_triangle(hit: RayIntersection) -> SceneHit {
    if hit.kind == RAY_QUERY_INTERSECTION_NONE { return scene_miss(); }
    if hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
        return scene_triangle(hit.t, hit.instance_custom_data, hit.primitive_index, hit.barycentrics, hit.front_face);
    }
    var invalid = scene_miss();
    invalid.kind = SCENE_HIT_INVALID;
    return invalid;
}
fn trace_ray_impl(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32, include_glass: bool) -> SceneHit {
    return scene_hit_from_triangle(trace_triangle_ray_impl(ray_origin, ray_direction, ray_t_min, ray_t_max, ray_flag, include_glass));
}
fn trace_triangle_ray_impl(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32, include_glass: bool) -> RayIntersection {
    let minimum = max(ray_t_min, max(max(abs(ray_origin.x), abs(ray_origin.y)), abs(ray_origin.z)) * sky_light.relative_ray_min);
    let maximum = min(ray_t_max, ray_max_distance());
    if minimum > maximum {
        var miss: RayIntersection;
        return miss;
    }
    let ray = RayDesc(ray_flag, RAY_NO_CULL, minimum, maximum, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    // Opaque geometry commits in hardware and never shows up here. Meshes
    // with an alpha-masked or blended material are built non-opaque, and each
    // of their hits is a candidate the shader confirms or drops. Glass is
    // committed only by the explicit glass-aware traversal.
    while rayQueryProceed(&rq) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        if candidate.kind == RAY_QUERY_INTERSECTION_TRIANGLE && candidate_is_solid(candidate, include_glass, ray_origin, ray_direction) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    return rayQueryGetCommittedIntersection(&rq);
}

// Alpha test for a candidate hit on non-opaque geometry.
fn candidate_is_solid(candidate: RayIntersection, include_glass: bool, origin: vec3f, direction: vec3f) -> bool {
    let material = materials[material_ids[candidate.instance_custom_data]];
    let glass = (material.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u;
    if glass && !include_glass { return false; }
    let diffuse = (material.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u;
    if !glass && !diffuse && (material.flags & MATERIAL_FLAG_ALPHA_MASK) == 0u { return true; }
    let barycentrics = vec3(1.0 - candidate.barycentrics.x - candidate.barycentrics.y, candidate.barycentrics);
    let vertices = load_vertices(geometry_ids[candidate.instance_custom_data], candidate.primitive_index);
    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;
    let alpha = resolve_material_alpha(material, uv);
    if diffuse {
        // Camera/glossy paths resolve coverage themselves. Other callers use
        // stochastic alpha with a stable hash of the ray and instance (PBRT style).
        if include_glass { return alpha > 0.0; }
        let o = bitcast<vec3u>(origin);
        let d = bitcast<vec3u>(direction);
        var h = o.x ^ (o.y * 1664525u) ^ (o.z * 1013904223u)
            ^ d.x ^ (d.y * 2246822519u) ^ (d.z * 3266489917u)
            ^ (candidate.instance_custom_data * 374761393u);
        h = (h ^ (h >> 16u)) * 2246822519u;
        h = (h ^ (h >> 13u)) * 3266489917u;
        h = h ^ (h >> 16u);
        return f32(h >> 8u) * (1.0 / 16777216.0) < clamp(alpha, 0.0, 1.0);
    }
    // Zero-coverage glass is a hole, and must not consume a path interaction.
    return select(alpha >= material.alpha_cutoff, alpha > 0.0, glass);
}

fn resolve_material_alpha(material: Material, uv: vec2<f32>) -> f32 {
    var alpha = material.base_color_alpha;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        alpha *= sample_texture_alpha(material.base_color_texture_id, uv);
    }
    return alpha;
}

// Diameter in metres plus diameter growth per metre, not a radius.
fn primary_ray_cone(projection: mat4x4f, pixels: vec2f, distance: f32) -> vec2f {
    let slope = max(2.0 / max(abs(projection[0][0]) * pixels.x, 1.0),
                    2.0 / max(abs(projection[1][1]) * pixels.y, 1.0));
    return vec2f(max(distance, 0.0) * slope, slope);
}
fn advance_ray_cone(cone: vec2f, distance: f32) -> vec2f {
    return vec2f(abs(cone.x + max(distance, 0.0) * cone.y), cone.y);
}
// Inverse transpose without assuming orthonormal or uniformly scaled instances.
fn ray_normal_matrix(m: mat3x3f) -> mat3x3f {
    let cofactor = mat3x3f(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    let determinant = dot(m[0], cofactor[0]);
    return cofactor * select(-1.0, 1.0, determinant >= 0.0);
}
fn ray_inverse_scale_bound(m: mat3x3f) -> f32 {
    let c0 = cross(m[1],m[2]); let c1 = cross(m[2],m[0]); let c2 = cross(m[0],m[1]);
    // Frobenius norm bounds the inverse's largest singular value, including shear.
    return sqrt(dot(c0,c0)+dot(c1,c1)+dot(c2,c2)) / max(abs(dot(m[0],c0)), 1e-20);
}
fn scatter_ray_cone(cone: vec2f, normal_spread: f32, roughness: f32) -> vec2f {
    return vec2f(cone.x, max(cone.y + 2.0 * normal_spread, 2.0 * roughness));
}
fn ray_texture_lod(id: u32, uv_width: vec2f) -> f32 {
    let texels = uv_width * vec2f(textureDimensions(textures[id], 0));
    return clamp(log2(max(max(texels.x, texels.y), 1.0)), 0.0,
        f32(textureNumLevels(textures[id]) - 1u));
}
fn sample_texture_filtered(id: u32, uv: vec2f, width: vec2f) -> vec3f {
    return textureSampleLevel(textures[id], samplers[id], uv, ray_texture_lod(id, width)).rgb;
}
// A triangle's two UV gradients in world space, including nonuniform scale.
fn ray_uv_width(vertices: array<Vertex,3>, positions: array<vec3f,3>, width: f32) -> vec2f {
    let e0 = positions[1] - positions[0];
    let e1 = positions[2] - positions[0];
    let n = cross(e0, e1);
    let inverse_area_squared = 1.0 / max(dot(n,n), 1e-30);
    let a = cross(e1,n) * inverse_area_squared;
    let b = cross(n,e0) * inverse_area_squared;
    let u = vertices[1].uv - vertices[0].uv;
    let v = vertices[2].uv - vertices[0].uv;
    return width * vec2f(length(a*u.x + b*v.x), length(a*u.y + b*v.y));
}
fn sample_texture(id: u32, uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).rgb; // TODO: Mipmap
}

fn sample_texture_alpha(id: u32, uv: vec2<f32>) -> f32 {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).a;
}

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
    reflectance: f32,
    perceptual_roughness: f32,
    roughness: f32,
    metallic: f32,
    diffuse_transmission: f32,
    emission_cone: vec4<f32>,
    gaussian_weight: f32,
    lommel: bool,
}

fn emitted_radiance(material: ResolvedMaterial, outgoing: vec3<f32>) -> vec3<f32> {
    return material.emissive * collimated_weight(material.emission_cone, outgoing);
}

struct ResolvedRayHitFull {
    world_position: vec3<f32>,
    previous_frame_world_position: vec3<f32>,
    world_normal: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    world_tangent: vec4<f32>,
    uv: vec2<f32>,
    triangle_area: f32,
    // Actual triangle normal for area/solid-angle Jacobians, never normal-mapped.
    triangle_world_normal: vec3<f32>,
    triangle_count: u32,
    light_probability: f32,
    material: ResolvedMaterial,
#ifdef RAY_MATERIAL_FOOTPRINTS
    ray_cone: vec2f,
    normal_spread: f32,
#endif
}

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    return resolve_material_filtered(material, uv, vec2f(0.0));
}
fn resolve_material_filtered(material: Material, uv: vec2f, uv_width: vec2f) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture_filtered(material.base_color_texture_id, uv, uv_width);
    }

    m.emission_cone = material.emission_cone;
    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture_filtered(material.emissive_texture_id, uv, uv_width);
    }

    m.reflectance = material.reflectance;
    m.lommel = (material.flags & 32u) != 0u;
    // Realtime estimators opt in only once both hemispheres are supported.
#ifdef FOLIAGE_TRANSMISSION
    m.diffuse_transmission = f32(material.flags >> 16u) / 65535.0;
#else
    m.diffuse_transmission = 0.0;
#endif

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = sample_texture_filtered(material.metallic_roughness_texture_id, uv, uv_width);
        m.perceptual_roughness *= metallic_roughness.g;
        m.metallic *= metallic_roughness.b;
    }

    m.gaussian_weight = 0.0;
    if material.gaussian.z > 0.0 {
        m.gaussian_weight = sample_texture_alpha(u32(material.gaussian.w), uv);
        m.reflectance = mix(m.reflectance, material.gaussian.x, m.gaussian_weight);
        m.perceptual_roughness = mix(m.perceptual_roughness, material.gaussian.y, m.gaussian_weight);
    }
    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    return m;
}

fn resolve_ray_hit_full(ray_hit: SceneHit) -> ResolvedRayHitFull {
    return resolve_ray_hit_filtered(ray_hit, vec3f(0.0), vec2f(0.0));
}
// Unknown tags never address triangle or material storage.
// This scene-level policy must include every representation when coarse events publish.
fn scene_requires_ordered_transport() -> bool {
    return (sky_light.material_transport_flags & (MATERIAL_FLAG_ALPHA_BLEND | MATERIAL_FLAG_DIFFUSE_BLEND)) != 0u;
}
fn scene_hit_material(ray_hit: SceneHit) -> Material {
    if !scene_hit_is_triangle(ray_hit) { var invalid: Material; return invalid; }
    return materials[material_ids[ray_hit.triangle.slot]];
}
fn scene_hit_alpha(ray_hit: SceneHit, uv: vec2f) -> f32 {
    if !scene_hit_is_triangle(ray_hit) { return 0.0; }
    return resolve_material_alpha(scene_hit_material(ray_hit), uv);
}
fn resolve_ray_hit_filtered(ray_hit: SceneHit, direction: vec3f, cone: vec2f) -> ResolvedRayHitFull {
    if !scene_hit_is_triangle(ray_hit) { var invalid: ResolvedRayHitFull; return invalid; }
    let barycentrics = vec3(1.0 - ray_hit.triangle.barycentrics.x - ray_hit.triangle.barycentrics.y, ray_hit.triangle.barycentrics);
    var hit = resolve_triangle_data_filtered(ray_hit.triangle.slot, ray_hit.triangle.primitive, barycentrics, direction, cone);
    // A double-sided surface hit from behind is shaded as the side the ray
    // arrived on (the rasteriser does the same for the gbuffer); a
    // single-sided back face keeps upstream's raw normal.
    let material = materials[material_ids[ray_hit.triangle.slot]];
    if !ray_hit.triangle.front_face && (material.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u {
        hit.world_normal = -hit.world_normal;
        hit.geometric_world_normal = -hit.geometric_world_normal;
    }
    return hit;
}

fn load_vertices(instance_geometry_ids: InstanceGeometryIds, triangle_id: u32) -> array<Vertex, 3> {
    let index_buffer = &index_buffers[instance_geometry_ids.index_buffer_id].indices;
    let vertex_buffer = &vertex_buffers[instance_geometry_ids.vertex_buffer_id].vertices;

    let indices_i = (triangle_id * 3u) + vec3(0u, 1u, 2u) + instance_geometry_ids.index_buffer_offset;
    let indices = vec3((*index_buffer)[indices_i.x], (*index_buffer)[indices_i.y], (*index_buffer)[indices_i.z]) + instance_geometry_ids.vertex_buffer_offset;

    return array<Vertex, 3>(
        unpack_vertex((*vertex_buffer)[indices.x]),
        unpack_vertex((*vertex_buffer)[indices.y]),
        unpack_vertex((*vertex_buffer)[indices.z])
    );
}

fn transform_positions(transform: mat4x4<f32>, vertices: array<Vertex, 3>) -> array<vec3<f32>, 3> {
    return array<vec3<f32>, 3>(
        (transform * vec4(vertices[0].position, 1.0)).xyz,
        (transform * vec4(vertices[1].position, 1.0)).xyz,
        (transform * vec4(vertices[2].position, 1.0)).xyz
    );
}

fn resolve_triangle_data_full(instance_id: u32, triangle_id: u32, barycentrics: vec3<f32>) -> ResolvedRayHitFull {
    return resolve_triangle_data_filtered(instance_id, triangle_id, barycentrics, vec3f(0.0), vec2f(0.0));
}
fn resolve_triangle_data_filtered(instance_id: u32, triangle_id: u32, barycentrics: vec3f, direction: vec3f, cone: vec2f) -> ResolvedRayHitFull {
    let material_id = material_ids[instance_id];
    let material = materials[material_id];

    let transform = transforms[instance_id];
    let previous_frame_transform = previous_frame_transforms[instance_id];

    let instance_geometry_ids = geometry_ids[instance_id];
    let vertices = load_vertices(instance_geometry_ids, triangle_id);

    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    let previous_frame_world_vertices = transform_positions(previous_frame_transform, vertices);
    let previous_frame_world_position = mat3x3(previous_frame_world_vertices[0], previous_frame_world_vertices[1], previous_frame_world_vertices[2]) * barycentrics;

    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;
    var footprint = 0.0;
#ifdef RAY_MATERIAL_FOOTPRINTS
    let face_normal = normalize(cross(world_vertices[1]-world_vertices[0], world_vertices[2]-world_vertices[0]));
    footprint = cone.x / max(abs(dot(face_normal, direction)), 0.0001);
#endif
    let uv_width = ray_uv_width(vertices, world_vertices, footprint);

    let object_linear = mat3x3f(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let normal_matrix = ray_normal_matrix(object_linear);
    let local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    let world_tangent = vec4(
        normalize(mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_tangent),
        vertices[0].tangent.w * select(-1.0, 1.0, determinant(object_linear) >= 0.0),
    );

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics; // TODO: Use barycentric lerp, ray_hit.object_to_world, cross product geo normal
    var world_normal = normalize(normal_matrix * local_normal);
    let geometric_world_normal = world_normal;
    var normal_variance = 0.0;
    var normal_spread = 0.0;
#ifdef RAY_MATERIAL_FOOTPRINTS
    let edge0 = world_vertices[1] - world_vertices[0];
    let edge1 = world_vertices[2] - world_vertices[0];
    let face = cross(edge0, edge1);
    let covector0 = cross(edge1,face) / max(dot(face,face),1e-30);
    let covector1 = cross(face,edge0) / max(dot(face,face),1e-30);
    let n0 = normalize(normal_matrix * vertices[0].normal);
    let dn0 = normalize(normal_matrix * vertices[1].normal) - n0;
    let dn1 = normalize(normal_matrix * vertices[2].normal) - n0;
    let curvature_squared = max(dot(covector0,covector0)*dot(dn0,dn0)
        + dot(covector1,covector1)*dot(dn1,dn1) + 2.0*dot(covector0,covector1)*dot(dn0,dn1), 0.0);
    normal_spread = sqrt(curvature_squared) * footprint;
    normal_variance = min(normal_spread*normal_spread/12.0, 1.0);
#endif
    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        var Nt = sample_texture_filtered(material.normal_map_texture_id, uv, uv_width) * 2.0 - 1.0;
#ifdef RAY_MATERIAL_FOOTPRINTS
        // Unrenormalized normal mips retain the first moment. Two-channel maps
        // retain the geometric variance fallback when their stored Z is constant.
        normal_variance += max(1.0-dot(Nt,Nt), 0.0)
            * min(ray_texture_lod(material.normal_map_texture_id, uv_width), 1.0);
#endif
        Nt.z = sqrt(max(1.0 - dot(Nt.xy, Nt.xy), 0.0)); // Reconstruct Z to support two-channel normal maps
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let triangle_edge0 = world_vertices[0] - world_vertices[1];
    let triangle_edge1 = world_vertices[0] - world_vertices[2];
    let triangle_area = length(cross(triangle_edge0, triangle_edge1)) / 2.0;

    var resolved_material = resolve_material_filtered(material, uv, uv_width);
    let detail = surface_details[material_id];
    if detail.textures.x != TEXTURE_MAP_NONE {
        let rotation = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
        let p = mat3x3(vertices[0].position, vertices[1].position, vertices[2].position) * barycentrics;
        let n = normalize(transpose(rotation) * world_normal);
        // Material coordinates retain their f64-derived phase. Convert the
        // world footprint to the material frame with an inverse-scale bound.
        let inverse_scale = ray_inverse_scale_bound(rotation);
        let sampled = sample_surface_detail(
            detail.textures.z, detail.textures.w, detail.textures.z,
            detail.meso.x, detail.meso.y,
            detail.coordinates, p, n, footprint * inverse_scale,
            textureSampleLevel(textures[detail.textures.x], samplers[detail.textures.x], uv, ray_texture_lod(detail.textures.x, uv_width)),
            textureSampleLevel(textures[detail.textures.y], samplers[detail.textures.y], uv, ray_texture_lod(detail.textures.y, uv_width)));
        resolved_material.base_color = clamp(resolved_material.base_color * sampled.colour, vec3f(0.0), vec3f(1.0));
        resolved_material.perceptual_roughness = clamp(resolved_material.perceptual_roughness * mix(sampled.roughness, 1.0, resolved_material.gaussian_weight), 0.02, 1.0);
        resolved_material.roughness = resolved_material.perceptual_roughness * resolved_material.perceptual_roughness;
        world_normal = normalize(normal_matrix * detail_normal(n, sampled.gradient));
    }
#ifdef RAY_MATERIAL_FOOTPRINTS
    // Add unresolved normal variance in GGX alpha-squared space.
    resolved_material.roughness = sqrt(min(1.0,
        resolved_material.roughness * resolved_material.roughness + normal_variance));
    resolved_material.perceptual_roughness = sqrt(resolved_material.roughness);
#endif
    // An exit aperture emits from the face toward its optical axis only.
    if material.emission_cone.w > 0.0 && dot(cross(triangle_edge0, triangle_edge1), material.emission_cone.xyz) <= 0.0 {
        resolved_material.emissive = vec3(0.0);
    }

    return ResolvedRayHitFull(
        world_position,
        previous_frame_world_position,
        world_normal,
        geometric_world_normal,
        world_tangent,
        uv,
        triangle_area,
        normalize(cross(triangle_edge0, triangle_edge1)),
        instance_geometry_ids.triangle_count,
        instance_geometry_ids.light_probability,
        resolved_material,
#ifdef RAY_MATERIAL_FOOTPRINTS
        cone,
        normal_spread,
#endif
    );
}
