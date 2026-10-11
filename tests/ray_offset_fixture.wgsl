@group(0) @binding(0) var<storage, read_write> output:array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let m=array<f32,8>(0.0,0.1,1.0,100.0,1e4,1e6,1e7,-1e7)[id.x%8u];
 var p=vec3(m);var n=normalize(vec3(1.0,2.0,3.0));
 var camera=p;
 var raster_error=0.0;
 if id.x>=32u && id.x<48u {
    let shift=f32(i32(id.x%3u)-1)*4096.0;
    p=vec3(1.15289306640625+shift,368.9375305175781-0.017425537109375,2.58612060546875-shift);
    n=vec3(0.0,1.0,0.0);
    camera=vec3(1.15289306640625+shift,2224.257080078125,9993.8515625-shift);
    // Actual captured GPU inverse projection, with translated camera column.
    // Exercise both near/far edges and native/double resolution at the measured
    // worst physical error. A correct origin must clear the floor, not a5cm lid.
    let inverse=mat4x4f(vec4f(0.7363797426,0.0,0.0,0.0),
        vec4f(0.0,0.407920748,-0.071927436,0.0),vec4f(camera,1.0),
        vec4f(-0.00011505932,-0.173396379,-0.984852135,0.0));
    let scale=1.0+f32(id.x%2u);
    let depth=select(0.0000883624,0.0001109711,(id.x%4u)>=2u);
    raster_error=raster_reconstruction_error(vec2u(vec2f(800.0,462.0)*scale),
        depth,n,vec2f(1600.0,900.0)*scale,inverse,SUBPIXEL_STEP);
 }
 if id.x>=48u {
    p=vec3(0.1,0.2,0.3);n=vec3(0.0,1.0,0.0);camera=p+vec3(0.0,0.0,1.0);
    // Orthographic and perspective-equivalent unit-depth footprint controls.
    let inverse=mat4x4f(vec4f(0.736,0.0,0.0,0.0),vec4f(0.0,0.414,0.0,0.0),
        vec4f(0.0,0.0,1.0,0.0),vec4f(0.0,0.0,0.0,1.0));
    raster_error=raster_reconstruction_error(vec2u(800u,450u),1.0,n,
        vec2f(1600.0,900.0),inverse,SUBPIXEL_STEP);
 }
 let moved=offset_surface_ray(p,n,camera,raster_error);
 output[id.x]=vec4(moved,dot(moved-p,n));
}
