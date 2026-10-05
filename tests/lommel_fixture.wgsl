@group(0) @binding(0) var<storage,read_write> output:array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    let nl=f32(id.x%4096u)/4096.0+0.0001;
    let nv=array<f32,3>(0.001,0.3,1.0)[id.x/4096u];
    let a=lommel_scattering(vec3(0.2),nl,nv).x;
    let b=lommel_scattering(vec3(0.2),nv,nl).x;
    output[id.x]=vec4(a,a/nl,b/nv,lommel_scattering(vec3(0.2),nl,nl).x);
}
