#define_import_path bevy_solarik::coarse_spatial_pack
fn spatial_sign(v: vec2f) -> vec2f { return select(vec2f(-1.0),vec2f(1.0),v>=vec2f(0.0)); }
fn spatial_pack(depth: f32, normal: vec3f, part: u32) -> u32 {
    var oct=normal.xy/dot(abs(normal),vec3f(1.0));
    if normal.z<0.0 { oct=(vec2f(1.0)-abs(oct.yx))*spatial_sign(oct); }
    let q=vec2u(round(clamp(oct*0.5+0.5,vec2f(0.0),vec2f(1.0))*511.0));
    return (u32(round(clamp(depth,0.0,1.0)*4094.0))+1u)|((part&3u)<<12u)|(q.x<<14u)|(q.y<<23u);
}
fn spatial_normal(word: u32) -> vec3f {
    let oct=vec2f(f32((word>>14u)&511u),f32(word>>23u))*(2.0/511.0)-1.0;
    var n=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
    if n.z<0.0 { let xy=(vec2f(1.0)-abs(n.yx))*spatial_sign(n.xy); n=vec3f(xy,n.z); }
    return normalize(n);
}
fn spatial_depth(word: u32) -> f32 { return (f32(word&4095u)-1.0)/4094.0; }
fn spatial_part(word: u32) -> u32 { return (word>>12u)&3u; }
fn spatial_axis(d: vec3f) -> u32 {
    let a=abs(d);
    if a.x>=a.y && a.x>=a.z { return 0u; }
    if a.y>=a.z { return 1u; }
    return 2u;
}
