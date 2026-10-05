enable wgpu_ray_query;

#define_import_path bevy_solarik::brdf
#import bevy_solarik::gaussian_math::{gaussian_specular, dielectric_ior, dielectric_fresnel}

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::lighting::{D_GGX, V_SmithGGXCorrelated, specular_multiscatter}
#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_pbr::utils::{rand_f, sample_cosine_hemisphere}
#import bevy_render::maths::{PI, orthonormalize}
#import bevy_solarik::sampling::{sample_ggx_vndf, ggx_vndf_pdf, ggx_vndf_sample_invalid}
#import bevy_solarik::scene_bindings::{ResolvedMaterial, MIRROR_ROUGHNESS_THRESHOLD, brdf_dfg_lut, brdf_dfg_lut_sampler}

struct EvaluateAndSampleBrdfResult {
    wi: vec3<f32>,
    throughput: vec3<f32>,
    pdf: f32,
}

// A surface point seen from `wo` faces `wo`. Interpolated and mapped shading
// normals can point past the viewer's horizon at grazing angles, where every
// lobe is zero: the silhouette of a normal-mapped surface is then outlined in
// black pixels. The facets such a normal stands for are hidden behind others
// that tilt towards the viewer about as far, so the normal is mirrored into
// the visible hemisphere. A normal that already faces `wo` is unchanged.
fn view_facing_normal(shading_normal: vec3<f32>, wo: vec3<f32>) -> vec3<f32> {
    let cosine = dot(shading_normal, wo);
    return shading_normal - 2.0 * min(cosine, 0.0) * wo;
}

fn evaluate_and_sample_brdf(
    wo: vec3<f32>,
    shading_normal: vec3<f32>,
    material: ResolvedMaterial,
    rng: ptr<function, u32>,
) -> EvaluateAndSampleBrdfResult {
    let world_normal = view_facing_normal(shading_normal, wo);
    let NdotV = dot(world_normal, wo);
    if NdotV < 0.0001 { return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0); }
    let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
    let df = 1.0 - luminance(specular_albedo(F0, NdotV, material.perceptual_roughness));

    let diffuse_weight = mix(df, 0.0, material.metallic);
    let specular_weight = 1.0 - diffuse_weight;

    let TBN = orthonormalize(world_normal);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));

    var wi: vec3<f32>;
    var wi_tangent: vec3<f32>;
    let diffuse_selected = rand_f(rng) < diffuse_weight;
    if diffuse_selected {
        // Preserve the old random sequence when transmission is disabled.
        var normal = world_normal;
        if material.diffuse_transmission > 0.0 {
            if rand_f(rng) < material.diffuse_transmission { normal = -normal; }
        }
        wi = sample_cosine_hemisphere(normal, rng);
        wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));
    } else {
        wi_tangent = sample_ggx_vndf(wo_tangent, material.roughness, rng);
        if ggx_vndf_sample_invalid(wi_tangent) {
            return EvaluateAndSampleBrdfResult(vec3(0.0), vec3(0.0), 0.0);
        }
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;
    }

    let pdf = evaluate_brdf_pdf(wo, wi, world_normal, material);
    if pdf <= 0.0 { return EvaluateAndSampleBrdfResult(wi, vec3(0.0), 0.0); }

    var throughput = evaluate_brdf(wo, wi, world_normal, material);
    if diffuse_selected || material.roughness > MIRROR_ROUGHNESS_THRESHOLD {
        throughput /= pdf;
    } else {
        throughput /= specular_weight;
    }

    return EvaluateAndSampleBrdfResult(wi, throughput, pdf);
}

fn evaluate_brdf(
    wo: vec3<f32>,
    wi: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
) -> vec3<f32> {
    return evaluate_diffuse_brdf(wo, wi, world_normal, material) + evaluate_specular_brdf(wo, wi, world_normal, material);
}

fn evaluate_diffuse_brdf(wo: vec3<f32>, wi: vec3<f32>, shading_normal: vec3<f32>, material: ResolvedMaterial) -> vec3<f32> {
    let world_normal = view_facing_normal(shading_normal, wo);
    let diffuse_color = calculate_diffuse_color(material.base_color, material.metallic, 0.0, 0.0) / PI;

    let NdotL = dot(world_normal, wi);
    let NdotV = dot(world_normal, wo);
    if abs(NdotL) < 0.0001 || NdotV < 0.0001 { return vec3(0.0); }
    let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
    // Light the specular layer reflects on the way in or out never reaches
    // the diffuse base.
    var layering = (1.0 - specular_albedo(F0, abs(NdotL), material.perceptual_roughness))
        * (1.0 - specular_albedo(F0, NdotV, material.perceptual_roughness));

    if material.gaussian_weight > 0.0 {
        let ior = dielectric_ior(material.reflectance);
        let transmission = (1.0-dielectric_fresnel(abs(NdotL),ior))*(1.0-dielectric_fresnel(NdotV,ior));
        layering = mix(layering, vec3(transmission), material.gaussian_weight);
    }
    let side_weight = select(1.0 - material.diffuse_transmission, material.diffuse_transmission, NdotL < 0.0);
    return diffuse_color * layering * abs(NdotL) * side_weight;
}

fn evaluate_specular_brdf(wo: vec3<f32>, wi: vec3<f32>, shading_normal: vec3<f32>, material: ResolvedMaterial) -> vec3<f32> {
    let world_normal = view_facing_normal(shading_normal, wo);
    if dot(world_normal, wi) < 0.0001 { return vec3(0.0); }
    let H = normalize(wi + wo);
    let NdotL = dot(world_normal, wi);
    let NdotH = dot(world_normal, H);
    let LdotH = dot(wi, H);
    let NdotV = dot(world_normal, wo);
    if NdotL < 0.0001 || NdotH < 0.0001 || LdotH < 0.0001 || NdotV < 0.0001 { return vec3(0.0); }

    let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
    let F = fresnel(F0, LdotH);

    if material.roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        if abs(NdotH - 1.0) < 0.0001 {
            return F;
        } else {
            return vec3(0.0);
        }
    }

    let D = D_GGX(material.roughness, NdotH);
    let Vs = V_SmithGGXCorrelated(material.roughness, NdotV, NdotL);
    let F_ab = F_AB(material.perceptual_roughness, NdotV);
    let ggx = specular_multiscatter(D, Vs, F, F0, F_ab, 1.0) * NdotL;
    if material.gaussian_weight <= 0.0 { return ggx; }
    let gaussian = gaussian_specular(material.roughness, dielectric_ior(material.reflectance), NdotV, NdotL, NdotH, LdotH);
    return mix(ggx, vec3(gaussian), material.gaussian_weight);
}

fn fresnel(f0: vec3<f32>, LdotH: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - LdotH, 5.0);
}

// Share of the light arriving at `cosine` to the normal that the specular
// layer reflects: the directional albedo of the GGX lobe from the split-sum
// table (Karis 2013, "Real Shading in Unreal Engine 4"). A smooth interface
// reflects its Fresnel reflectance, which the table reproduces at zero
// roughness. A rough one reflects far less at grazing angles; using the smooth
// value there removed nine tenths of the diffuse light from every rough
// surface seen edge-on.
fn specular_albedo(F0: vec3<f32>, cosine: f32, perceptual_roughness: f32) -> vec3<f32> {
    let F_ab = F_AB(perceptual_roughness, cosine);
    return saturate(F0 * F_ab.x + F_ab.y);
}

// Scale/bias approximation
fn F_AB(perceptual_roughness: f32, NdotV: f32) -> vec2<f32> {
    return textureSampleLevel(brdf_dfg_lut, brdf_dfg_lut_sampler, vec2<f32>(NdotV, perceptual_roughness), 0.0).rg;
}

// Solid-angle PDF shared by continuation sampling and light-sample MIS.
fn evaluate_brdf_pdf(wo: vec3<f32>, wi: vec3<f32>, shading_normal: vec3<f32>, material: ResolvedMaterial) -> f32 {
    let world_normal = view_facing_normal(shading_normal, wo);
    let NdotV = dot(world_normal, wo);
    if NdotV < 0.0001 { return 0.0; }
    let NdotL = dot(world_normal, wi);
    let F0 = calculate_F0(material.base_color, material.metallic, vec3(material.reflectance));
    let diffuse_weight = mix(1.0 - luminance(specular_albedo(F0, NdotV, material.perceptual_roughness)), 0.0, material.metallic);
    let side_weight = select(1.0 - material.diffuse_transmission, material.diffuse_transmission, NdotL < 0.0);
    let diffuse_pdf = diffuse_weight * side_weight * abs(NdotL) / PI;
    // GGX has support only on the reflection hemisphere.
    if NdotL <= 0.0 { return diffuse_pdf; }
    let TBN = orthonormalize(world_normal);
    let wo_tangent = transpose(TBN) * wo;
    let wi_tangent = transpose(TBN) * wi;
    return diffuse_pdf + (1.0 - diffuse_weight) * ggx_vndf_pdf(wo_tangent, wi_tangent, material.roughness);
}
