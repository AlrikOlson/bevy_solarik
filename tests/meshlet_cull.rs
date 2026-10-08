//! Actual Bevy composition for both production instance-cull variants.
//! Geometry gates and unused imported uniform types are narrow fixture dependencies;
//! native raster images/readbacks remain the acceptance for geometry and coverage.
use bevy_asset::{AssetId, Assets};
use bevy_shader::{Shader, ShaderCache, ShaderDefVal};

fn register(
    cache: &mut ShaderCache<(), ()>,
    assets: &mut Assets<Shader>,
    source: &str,
) -> AssetId<Shader> {
    let shader = Shader::from_wgsl(source.to_owned(), "instance-cull-test");
    let id = assets.add(shader.clone()).id();
    cache.set_shader(id, shader);
    id
}

fn dependencies(cache: &mut ShaderCache<(), ()>, assets: &mut Assets<Shader>) {
    for source in [
        "#define_import_path bevy_pbr::mesh_types\nstruct Mesh { value: vec4<f32> }",
        "#define_import_path bevy_render::view\nstruct View { value: vec4<f32> }",
        "#define_import_path bevy_pbr::prepass_bindings\nstruct PreviousViewUniforms { value: vec4<f32> }",
        "#define_import_path bevy_pbr::utils\nfn octahedral_decode_signed(v: vec2<f32>) -> vec3<f32> { return vec3(v, 1.0); }",
        include_str!("../vendor/bevy_pbr/src/meshlet/meshlet_bindings.wgsl"),
        "#define_import_path bevy_pbr::meshlet_cull_shared\n#import bevy_pbr::meshlet_bindings::MeshletAabb\nfn aabb_in_frustum(a: MeshletAabb, i: u32) -> bool { return true; }\nfn should_occlusion_cull_aabb(a: MeshletAabb, i: u32) -> bool { return false; }",
    ] {
        register(cache, assets, source);
    }
}

#[test]
fn production_meshlet_cull_composes_first_and_second_active_slot_variants() {
    let mut assets = Assets::<Shader>::default();
    let mut cache = ShaderCache::new(
        (),
        wgpu::Features::all(),
        wgpu::DownlevelFlags::all(),
        |_, _, _| Ok(()),
    );
    dependencies(&mut cache, &mut assets);
    let id = register(
        &mut cache,
        &mut assets,
        include_str!("../vendor/bevy_pbr/src/meshlet/cull_instances.wgsl"),
    );
    for first in [false, true] {
        let defs = [
            ShaderDefVal::Bool("MESHLET_INSTANCE_CULLING_PASS".into(), true),
            ShaderDefVal::Bool("MESHLET_FIRST_CULLING_PASS".into(), first),
        ];
        cache
            .get(usize::from(first), id, &defs)
            .expect("actual cull shader composition");
    }
}
