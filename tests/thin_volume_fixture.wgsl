@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x; let q=vec3(f32(i%4u)*1.5,0.2,-8.0);
 let v=select(vec3(0.3,0.4,0.5),vec3(0.0,0.0,0.5),i/16u==0u);
 let ends=array<f32,4>(0.0,8.0,16.0,40.0);
 result[i]=vec4(gaussian_segment(q,v,ends[(i/4u)%4u]),0.0,0.0,0.0);
}
