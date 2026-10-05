@group(0) @binding(0) var<storage,read_write> results:array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;let x=0.05+f32(i%8u)*0.7;let a=f32(i/8u)*0.4;let b=f32(7u-i/8u)*0.3;
 var p=vec3(1.5e8,1e8,0.0);var d=vec3(0.0,-1.0,0.0);
 if i%4u==1u {p=vec3(1.5e8,100.0,0.0);d=vec3(0.8,-0.6,0.0);}
 if i%4u==2u {p=vec3(1.5e8,0.0,0.0);d=vec3(1.0,0.0,0.0);}
 if i%4u==3u {p=vec3(1.5e8,6.0,0.0);d=vec3(1.0,0.0,0.0);}
 let column=ring_column(p,d);
 var height=0.0;
 if i%4u==1u {height=5e7;}
 if i%4u==2u {height=4.5e7;}
 if i%4u==3u {height=4.5e7+37500.0;}
 results[i]=vec4(ring_source(x,a,b),column,ring_planet_visibility(vec3(1.5e8,height,0.0),vec3(-1.0,0.0,0.0)),exp(-column));
}
