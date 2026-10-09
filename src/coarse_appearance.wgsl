#define_import_path bevy_solarik::coarse_appearance
struct CoarseMaterial { base_color: vec4f, surface: vec4f, optical: vec4f }
struct CoarseSurface { color: vec4f, normal: vec4f, surface: vec4f }
struct CoarseAppearanceMoment {
    color: vec4f, normal: vec4f, diagonal: vec4f, cross_moment: vec4f,
    surface: vec4f, counts: vec4u
}
// Same source frame convention as Solarik's ray material resolve: normalize
// interpolated N/T, preserve handedness, reconstruct tangent-space normal Z.
fn coarse_surface(material: CoarseMaterial, base: vec4f, normal_map: vec4f,
    metal_rough: vec4f, vertex_normal: vec3f, tangent: vec4f) -> CoarseSurface {
    let n=normalize(vertex_normal);
    let t=normalize(tangent.xyz);
    let b=tangent.w*cross(n,t);
    var nt=normal_map.xyz*2.0-1.0;
    nt.z=sqrt(max(1.0-dot(nt.xy,nt.xy),0.0));
    nt.y*=select(1.0,-1.0,material.optical.y!=0.0);
    let shading=normalize(t*nt.x+b*nt.y+n*nt.z);
    let roughness=material.surface.x*metal_rough.y;
    let r2=roughness*roughness;
    return CoarseSurface(material.base_color*base,vec4f(shading,0.0),
        vec4f(r2*r2,material.surface.y*metal_rough.z,material.surface.zw));
}
fn coarse_accumulate(moment: ptr<function,CoarseAppearanceMoment>, sample: CoarseSurface) {
    (*moment).color+=sample.color;
    (*moment).normal+=sample.normal;
    let n=sample.normal.xyz;
    (*moment).diagonal+=vec4f(n*n,0.0);
    (*moment).cross_moment+=vec4f(n.x*n.y,n.x*n.z,n.y*n.z,0.0);
    (*moment).surface+=sample.surface;
    (*moment).counts.x+=1u;
}
