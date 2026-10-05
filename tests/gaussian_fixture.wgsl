@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
 let wind=array<f32,3>(1.0,5.0,12.0)[id.x/4096u];
 let c=(f32(id.x%4096u)+0.5)/4096.0;
 let a=sqrt(0.003+0.00508*wind);
 output[id.x]=vec4(dielectric_fresnel(c,1.333),gaussian_ndf(a,c),
 gaussian_specular(a,1.333,0.3,0.8,c,0.75)*0.3,
 gaussian_specular(a,1.333,0.8,0.3,c,0.75)*0.8);
}
