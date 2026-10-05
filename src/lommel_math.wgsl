#define_import_path bevy_solarik::lommel_math
// BRDF times incoming cosine, the convention of Solarik light estimators.
fn lommel_scattering(rho:vec3<f32>,nl:f32,nv:f32)->vec3<f32> {
    if min(nl,nv)<=0.0 {return vec3(0.0);}
    return 2.0*clamp(rho,vec3(0.0),vec3(0.25))*nl/(3.141592653589793*(nl+nv));
}
fn lommel_gbuffer(gbuffer:vec4<u32>,enabled:f32)->vec4<u32> {
    var result=gbuffer;
    if enabled>0.0 {result.a|=0x10000000u;}
    return result;
}
