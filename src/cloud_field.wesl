// Shared weather geography for visible clouds and surface/air shadows.
// Each belt rigidly advects a spherical field. Blend occupied column moments
// at the boundary; multiplying wrapped angles by weights would jump at 2 pi.
struct WeatherCoordinates {
    departure: array<vec3<f32>,3>,
    weights: vec3<f32>,
}
fn weather_departure(unit:vec3<f32>,phase:f32)->vec3<f32> {
    let c=cos(phase); let s=sin(phase);
    return vec3(c*unit.x+s*unit.z,unit.y,-s*unit.x+c*unit.z);
}
fn weather_coordinates(unit:vec3<f32>,motion:array<vec4<f32>,2>)->WeatherCoordinates {
    if motion[1].z<=0.0 || all(motion[0].xyz==vec3(0.0)) {
        return WeatherCoordinates(array<vec3<f32>,3>(unit,unit,unit),vec3(1.0,0.0,0.0));
    }
    let latitude=asin(clamp(abs(unit.y),0.0,1.0));
    let edges=motion[1].xy; let width=motion[1].z;
    let transition=smoothstep(edges-vec2(width),edges+vec2(width),vec2(latitude));
    return WeatherCoordinates(array<vec3<f32>,3>(
        weather_departure(unit,motion[0].x),weather_departure(unit,motion[0].y),
        weather_departure(unit,motion[0].z)),vec3(1.0-transition.x,transition.x-transition.y,transition.y));
}
// R cover, G top, B extinction, A thickness/top, all normalized to the shell.
fn weather_moments(v:vec4<f32>)->vec4<f32> {
    return vec4(v.r,v.r*vec3(v.g,v.g*v.a,v.g*v.a*v.b));
}
fn weather_from_moments(v:vec4<f32>)->vec4<f32> {
    if v.x<=1e-8 {return vec4(0.0);}
    let top=v.y/v.x; let thickness=v.z/v.x;
    return vec4(v.x,top,v.w/max(v.z,1e-8),thickness/max(top,1e-8));
}
fn weather_sample(weather:texture_cube<f32>,filtering:sampler,
    coordinates:WeatherCoordinates,lod:f32)->vec4<f32> {
    var moments=vec4(0.0);
    for(var i=0u;i<3u;i++) {
        if coordinates.weights[i]>0.0 {
            let v=textureSampleLevel(weather,filtering,coordinates.departure[i],lod);
            moments+=coordinates.weights[i]*weather_moments(v);
        }
    }
    return weather_from_moments(moments);
}
fn weather_profile_above(f:f32)->f32 {
    let u=clamp((f-0.85)/0.15,0.0,1.0);
    return 0.15*(0.5-u+u*u*u-0.5*u*u*u*u)
        +0.51*(1.0-pow(clamp(f/0.85,0.0,1.0),5.0/3.0));
}
fn weather_sphere_exit(p:vec3<f32>,direction:vec3<f32>,radius:f32)->f32 {
    let b=dot(p,direction); let perpendicular=p-b*direction;
    return -b+sqrt(max(radius*radius-dot(perpendicular,perpendicular),0.0));
}
// Result: cover, density-weighted physical column length, source zenith cosine.
// Radii/positions use the caller's unit (metres for scene, km for atmosphere).
fn weather_shadow_column(p:vec3<f32>,direction:vec3<f32>,base:f32,top:f32,
    motion:array<vec4<f32>,2>,weather:texture_cube<f32>,filtering:sampler)->vec3<f32> {
    let r=length(p);
    if r>=top {return vec3(0.0);}
    let at_base=p+direction*select(0.0,weather_sphere_exit(p,direction,base),r<base);
    let mu=dot(normalize(at_base),direction);
    if mu<=0.0 {return vec3(0.0);}
    let low=weather_sample(weather,filtering,weather_coordinates(normalize(at_base),motion),0.0);
    let shell=top-base;
    let middle=base+low.g*shell*(1.0-0.5*low.a);
    let at_middle=p+direction*select(0.0,weather_sphere_exit(p,direction,middle),r<middle);
    let high=weather_sample(weather,filtering,weather_coordinates(normalize(at_middle),motion),0.0);
    let v=weather_from_moments(0.5*(weather_moments(low)+weather_moments(high)));
    let thickness=v.g*v.a*shell;
    let local_base=base+v.g*shell-thickness;
    let above=weather_profile_above((r-local_base)/max(thickness,1e-6));
    return vec3(v.r,v.b*thickness*above,mu);
}
