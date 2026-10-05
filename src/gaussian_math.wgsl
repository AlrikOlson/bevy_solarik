#define_import_path bevy_solarik::gaussian_math

// PBRT4 Fresnel equations, incident air and nonabsorbing dielectric.
fn dielectric_fresnel(cosine: f32, ior: f32) -> f32 {
    if ior == 1.0 { return 0.0; }
    let c = clamp(cosine, 0.0, 1.0);
    let t = sqrt(max(0.0, 1.0 - (1.0-c*c)/(ior*ior)));
    let rs = (c-ior*t)/max(c+ior*t, 1e-8);
    let rp = (ior*c-t)/max(ior*c+t, 1e-8);
    return 0.5*(rs*rs+rp*rp);
}
fn dielectric_ior(reflectance: f32) -> f32 {
    let r = clamp(0.4*reflectance, 0.0, 0.999);
    return (1.0+r)/(1.0-r);
}
// Heitz 2014: alpha² is TOTAL Gaussian slope variance.
fn gaussian_ndf(alpha: f32, cosine: f32) -> f32 {
    if cosine <= 0.0 { return 0.0; }
    let c2 = cosine*cosine;
    let a2 = alpha*alpha;
    return exp(-(1.0-c2)/(a2*c2))/(3.141592653589793*a2*c2*c2);
}
// Heitz Eq69 rational approximation to Gaussian Smith Lambda.
fn gaussian_lambda(alpha: f32, cosine: f32) -> f32 {
    let c = clamp(cosine, 1e-6, 1.0);
    let a = c/(alpha*sqrt(max(1e-12, 1.0-c*c)));
    if a >= 1.6 { return 0.0; }
    return max(0.0, (1.0-1.259*a+0.396*a*a)/(3.535*a+2.181*a*a));
}
// BRDF multiplied by incoming cosine, as required by Solarik estimators.
fn gaussian_specular(alpha: f32, ior: f32, nv: f32, nl: f32, nh: f32, lh: f32) -> f32 {
    if min(nv,nl) <= 0.0 { return 0.0; }
    let g = 1.0/(1.0+gaussian_lambda(alpha,nv)+gaussian_lambda(alpha,nl));
    return dielectric_fresnel(lh,ior)*gaussian_ndf(alpha,nh)*g/(4.0*nv);
}
// Bit3 of the deferred flags marks props.a as Gaussian coverage, not clearcoat.
fn gaussian_gbuffer(gbuffer: vec4<u32>, weight: f32) -> vec4<u32> {
    var out = gbuffer;
    let props = unpack4x8unorm(out.b);
    out.b = pack4x8unorm(vec4(props.rgb, weight));
    out.a |= 0x08000000u;
    return out;
}
