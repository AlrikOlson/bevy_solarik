@group(0) @binding(0) var<storage, read_write> output:array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let m=array<f32,8>(0.0,0.1,1.0,100.0,1e4,1e6,1e7,-1e7)[id.x%8u];
 let p=vec3(m);let n=normalize(vec3(1.0,2.0,3.0));
 let moved=offset_surface_ray(p,n);
 output[id.x]=vec4(moved,dot(moved-p,n));
}
