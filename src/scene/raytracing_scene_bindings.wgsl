enable wgpu_ray_query;

#define_import_path bevy_solarik::scene_bindings

#import bevy_pbr::lighting::perceptualRoughnessToRoughness
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace

// Scalar addressing preserves every allocator stride, including UV1 and colour.
struct InstanceGeometryIds {
    vertex_buffer_offset: u32, index_buffer_offset: u32,
    vertex_stride: u32, material_id: u32,
    normal_offset: u32, uv0_offset: u32, uv1_offset: u32, tangent_offset: u32,
    colour_offset: u32, triangle_count: u32, light_probability: f32, _padding: u32,
}
struct InstanceTransforms { current: mat4x4f, previous: mat4x4f }
struct Vertex {
    position: vec3f, normal: vec3f, uv: vec2f, tangent: vec4f,
    uv1: vec2f, colour: vec4f,
}
struct SceneMisc {
    intensity: f32,
    light_count: u32,
    previous_light_ids: array<u32>,
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
    _padding: vec3<f32>,
}

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    cos_theta_max: f32,
    luminance: vec3<f32>,
    inverse_pdf: f32,
}

const LIGHT_NOT_PRESENT_THIS_FRAME = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage> geometry_page_0: array<u32>;
@group(0) @binding(1) var<storage> geometry_page_1: array<u32>;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var samplers: binding_array<sampler>;
@group(0) @binding(4) var<storage> materials: array<Material>;
@group(0) @binding(5) var tlas: acceleration_structure;
@group(0) @binding(6) var<storage> instance_transforms: array<InstanceTransforms>;
@group(0) @binding(7) var<storage> geometry_ids: array<InstanceGeometryIds>;
@group(0) @binding(8) var<storage> light_sources: array<LightSource>;
@group(0) @binding(9) var<storage> directional_lights: array<DirectionalLight>;
@group(0) @binding(10) var<storage> local_lights: array<LocalLight>;
@group(0) @binding(11) var<storage> scene_misc: SceneMisc;
@group(0) @binding(12) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(13) var brdf_dfg_lut_sampler: sampler;
@group(0) @binding(14) var sky_texture: texture_cube<f32>;
@group(0) @binding(15) var sky_sampler: sampler;

fn instance_material_id(instance: u32) -> u32 { return geometry_ids[instance].material_id; }
fn translate_previous_light(light: u32) -> u32 {
    if light >= arrayLength(&scene_misc.previous_light_ids) { return LIGHT_NOT_PRESENT_THIS_FRAME; }
    return scene_misc.previous_light_ids[light];
}
fn scene_light_count() -> u32 { return scene_misc.light_count; }

// Radiance arriving from the sky along `direction` (world space, pointing
// away from the surface), for a ray that left the scene. Black when the
// scene has no sky. Cube maps are left-handed, so z is negated the way
// Bevy's skybox and environment map shaders do it.
fn sample_sky(direction: vec3<f32>) -> vec3<f32> {
    let cube_direction = vec3(direction.xy, -direction.z);
    return textureSampleLevel(sky_texture, sky_sampler, cube_direction, 0.0).rgb * scene_misc.intensity;
}

const RAY_T_MIN = 0.001f;
const RAY_T_MAX = 100000.0f;

const RAY_NO_CULL = 0xFFu;

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32) -> RayIntersection {
    return trace_ray_impl(ray_origin, ray_direction, ray_t_min, ray_t_max, ray_flag, false);
}

// Camera and pathtracer bounce rays can interact with glass. Shadow and
// realtime GI callers retain trace_ray's transparent-to-light behavior.
fn trace_glass_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32) -> RayIntersection {
    return trace_ray_impl(ray_origin, ray_direction, ray_t_min, ray_t_max, RAY_FLAG_NONE, true);
}

fn trace_ray_impl(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32, ray_flag: u32, include_glass: bool) -> RayIntersection {
#ifdef SOLARIK_METAL_ALPHA
    // Naga 29's Metal intersector has no candidate-confirm operation. Trace
    // nearest opaque hits and re-cast past rejected coverage instead.
    // Advance beyond intersection precision; bound work for dense foliage.
    // Vulkan retains the native candidate traversal below.
    var next_t = max(ray_t_min, 0.0);
    for (var layer = 0u; layer < 128u; layer += 1u) {
        var rq: ray_query;
        let ray = RayDesc(RAY_FLAG_FORCE_OPAQUE, RAY_NO_CULL, next_t, ray_t_max, ray_origin, ray_direction);
        rayQueryInitialize(&rq, tlas, ray);
        // Naga's Metal intersector completes during initialize; proceed's ready
        // flag never clears in Naga 29. Calling it in a loop hangs the GPU.
        rayQueryProceed(&rq);
        let hit = rayQueryGetCommittedIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE || candidate_is_solid(hit, include_glass, ray_origin, ray_direction) {
            return hit;
        }
        // A saturated stack conservatively occludes instead of hanging the GPU.
        if layer == 127u { return hit; }
        next_t = max(next_t, hit.t) + max(0.00001, abs(hit.t) * 0.000001);
        if next_t >= ray_t_max {
            var miss: RayIntersection;
            return miss;
        }
    }
    var miss: RayIntersection;
    return miss;
#else
    let ray = RayDesc(ray_flag, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
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
#endif
}

// Alpha test for a candidate hit on non-opaque geometry.
fn candidate_is_solid(candidate: RayIntersection, include_glass: bool, origin: vec3f, direction: vec3f) -> bool {
    let material = materials[instance_material_id(candidate.instance_index)];
    let glass = (material.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u;
    if glass && !include_glass { return false; }
    let diffuse = (material.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u;
    if !glass && !diffuse && (material.flags & MATERIAL_FLAG_ALPHA_MASK) == 0u { return true; }
    let barycentrics = vec3(1.0 - candidate.barycentrics.x - candidate.barycentrics.y, candidate.barycentrics);
    let vertices = load_vertices(geometry_ids[candidate.instance_index], candidate.primitive_index);
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
            ^ (candidate.instance_index * 374761393u);
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
}

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture(material.base_color_texture_id, uv);
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture(material.emissive_texture_id, uv);
    }

    m.reflectance = material.reflectance;
    // Realtime estimators opt in only once both hemispheres are supported.
#ifdef FOLIAGE_TRANSMISSION
    m.diffuse_transmission = f32(material.flags >> 16u) / 65535.0;
#else
    m.diffuse_transmission = 0.0;
#endif

    m.perceptual_roughness = material.perceptual_roughness;
    m.metallic = material.metallic;
    if material.metallic_roughness_texture_id != TEXTURE_MAP_NONE {
        let metallic_roughness = sample_texture(material.metallic_roughness_texture_id, uv);
        m.perceptual_roughness *= metallic_roughness.g;
        m.metallic *= metallic_roughness.b;
    }

    m.roughness = m.perceptual_roughness * m.perceptual_roughness;

    return m;
}

fn resolve_ray_hit_full(ray_hit: RayIntersection) -> ResolvedRayHitFull {
    let barycentrics = vec3(1.0 - ray_hit.barycentrics.x - ray_hit.barycentrics.y, ray_hit.barycentrics);
    var hit = resolve_triangle_data_full(ray_hit.instance_index, ray_hit.primitive_index, barycentrics);
    // A double-sided surface hit from behind is shaded as the side the ray
    // arrived on (the rasteriser does the same for the gbuffer); a
    // single-sided back face keeps upstream's raw normal.
    let material = materials[instance_material_id(ray_hit.instance_index)];
    if !ray_hit.front_face && (material.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u {
        hit.world_normal = -hit.world_normal;
        hit.geometric_world_normal = -hit.geometric_world_normal;
    }
    return hit;
}

// One canonical word address space across the two bounded Metal bindings.
fn geometry_word(at: u32) -> u32 {
    let split = arrayLength(&geometry_page_0);
    if at < split { return geometry_page_0[at]; }
    return geometry_page_1[at - split];
}
fn vertex_word(at: u32) -> f32 {
    return bitcast<f32>(geometry_word(at));
}
fn vertex2(at: u32) -> vec2f {
    return vec2(vertex_word(at), vertex_word(at + 1u));
}
fn vertex3(at: u32) -> vec3f {
    return vec3(vertex_word(at), vertex_word(at + 1u), vertex_word(at + 2u));
}
fn vertex4(at: u32) -> vec4f {
    return vec4(vertex3(at), vertex_word(at + 3u));
}
fn load_vertex(g: InstanceGeometryIds, i: u32) -> Vertex {
    let at = g.vertex_buffer_offset + i * g.vertex_stride;
    var v: Vertex;
    v.position = vertex3(at);
    v.normal = vertex3(at + g.normal_offset);
    v.uv = vertex2(at + g.uv0_offset);
    v.tangent = vertex4(at + g.tangent_offset);
    v.uv1 = vec2(0.0);
    v.colour = vec4(1.0);
    if g.uv1_offset != 0xFFFFFFFFu { v.uv1 = vertex2(at + g.uv1_offset); }
    if g.colour_offset != 0xFFFFFFFFu { v.colour = vertex4(at + g.colour_offset); }
    return v;
}
fn load_vertices(g: InstanceGeometryIds, triangle_id: u32) -> array<Vertex, 3> {
    let at = g.index_buffer_offset + triangle_id * 3u;
    return array(load_vertex(g, geometry_word(at)), load_vertex(g, geometry_word(at + 1u)),
        load_vertex(g, geometry_word(at + 2u)));
}

fn transform_positions(transform: mat4x4<f32>, vertices: array<Vertex, 3>) -> array<vec3<f32>, 3> {
    return array<vec3<f32>, 3>(
        (transform * vec4(vertices[0].position, 1.0)).xyz,
        (transform * vec4(vertices[1].position, 1.0)).xyz,
        (transform * vec4(vertices[2].position, 1.0)).xyz
    );
}

fn resolve_triangle_data_full(instance_id: u32, triangle_id: u32, barycentrics: vec3<f32>) -> ResolvedRayHitFull {
    let material_id = instance_material_id(instance_id);
    let material = materials[material_id];

    let transform = instance_transforms[instance_id].current;
    let previous_frame_transform = instance_transforms[instance_id].previous;

    let instance_geometry_ids = geometry_ids[instance_id];
    let vertices = load_vertices(instance_geometry_ids, triangle_id);

    let world_vertices = transform_positions(transform, vertices);
    let world_position = mat3x3(world_vertices[0], world_vertices[1], world_vertices[2]) * barycentrics;

    let previous_frame_world_vertices = transform_positions(previous_frame_transform, vertices);
    let previous_frame_world_position = mat3x3(previous_frame_world_vertices[0], previous_frame_world_vertices[1], previous_frame_world_vertices[2]) * barycentrics;

    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;

    let local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
    let world_tangent = vec4(
        normalize(mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_tangent),
        vertices[0].tangent.w,
    );

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics; // TODO: Use barycentric lerp, ray_hit.object_to_world, cross product geo normal
    var world_normal = normalize(mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_normal);
    let geometric_world_normal = world_normal;
    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let TBN = calculate_tbn_mikktspace(world_normal, world_tangent);
        let T = TBN[0];
        let B = TBN[1];
        let N = TBN[2];
        var Nt = sample_texture(material.normal_map_texture_id, uv) * 2.0 - 1.0;
        Nt.z = sqrt(max(1.0 - dot(Nt.xy, Nt.xy), 0.0)); // Reconstruct Z to support two-channel normal maps
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let triangle_edge0 = world_vertices[0] - world_vertices[1];
    let triangle_edge1 = world_vertices[0] - world_vertices[2];
    let triangle_area = length(cross(triangle_edge0, triangle_edge1)) / 2.0;

    var resolved_material = resolve_material(material, uv);
    let colour = mat3x3(vertices[0].colour.rgb, vertices[1].colour.rgb, vertices[2].colour.rgb) * barycentrics;
    resolved_material.base_color *= colour;

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
    );
}
