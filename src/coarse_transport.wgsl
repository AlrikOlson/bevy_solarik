#define_import_path bevy_solarik::coarse_transport
// Homogeneous surrogate only. Finite measurements do not prove source transport.
struct CoarseKernel { rates: array<f32,14>, measured: u32, reserved: u32 }
struct CoarseDirection { indices: vec3u, weights: vec3f }
struct CoarseRate { value: f32, unresolved: u32 }
// Positive piecewise-linear interpolation on the 24 axis/corner triangles.
fn coarse_transport_direction(d: vec3f) -> CoarseDirection {
    let a=abs(d); let m=min(a.x,min(a.y,a.z));
    var first=0u; var second=1u;
    if a.x<=a.y && a.x<=a.z { first=1u; second=2u; }
    else if a.y<=a.z { first=0u; second=2u; }
    var corner=6u;
    if d.x*d.y*d.z<0.0 {
        if d.x<0.0 && d.y<0.0 && d.z<0.0 { corner=7u; }
        else if d.x<0.0 { corner=12u; }
        else if d.y<0.0 { corner=10u; }
        else { corner=8u; }
    } else {
        if d.y<0.0 && d.z<0.0 { corner=13u; }
        else if d.x<0.0 && d.z<0.0 { corner=11u; }
        else if d.x<0.0 && d.y<0.0 { corner=9u; }
    }
    let weights=vec3f(a[first]-m,a[second]-m,m*sqrt(3.0));
    return CoarseDirection(vec3u(first*2u+u32(d[first]<0.0),
        second*2u+u32(d[second]<0.0),corner),weights/dot(weights,vec3f(1.0)));
}
fn coarse_transport_rate(kernel: CoarseKernel, basis: CoarseDirection) -> CoarseRate {
    var result: CoarseRate;
    for(var i=0u;i<3u;i+=1u) {
        let weight=basis.weights[i];
        if weight<=0.0 { continue; }
        let index=basis.indices[i];
        result.value+=weight*kernel.rates[index];
        if (kernel.measured&(1u<<index))==0u { result.unresolved=1u; }
    }
    return result;
}
fn coarse_transport_coverage(optical_depth: f32) -> f32 {
    if optical_depth<0.01 {
        return optical_depth*(1.0-optical_depth*(0.5-optical_depth/6.0));
    }
    return 1.0-exp(-optical_depth);
}
// Conditional mean collision distance within a homogeneous interval.
fn coarse_transport_centroid(rate: f32, length: f32) -> f32 {
    let x=rate*length;
    if x<0.02 { return length*(0.5-x/12.0+x*x*x/720.0); }
    let transmission=exp(-x);
    return 1.0/rate-length*transmission/(1.0-transmission);
}
struct CoarseFreeFlight { distance: f32, event: u32, valid: u32 }
// One exponential optical-depth budget carries unchanged across clipped cells.
// u is in [0,1), rate is in reciprocal ray-distance units.
fn coarse_free_flight(rate: f32, range: vec2f, u: f32) -> CoarseFreeFlight {
    var result: CoarseFreeFlight;
    if !(rate>=0.0 && rate<1e20 && range.x>=0.0 && range.y>range.x && range.y<1e20
        && u>=0.0 && u<1.0) { return result; }
    result.valid=1u;
    if rate==0.0 { return result; }
    let offset=-log(1.0-u)/rate;
    if offset<range.y-range.x { result.distance=range.x+offset; result.event=1u; }
    return result;
}
