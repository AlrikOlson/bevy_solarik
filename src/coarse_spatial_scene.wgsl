#define_import_path bevy_solarik::coarse_spatial_scene
const SPATIAL_MISSING=0xffffffffu;
// Exact rank among two 32-bit occupancy halves; never shift by 32.
fn spatial_sample_index(row: vec4u, texel: u32) -> u32 {
    if texel>=64u { return SPATIAL_MISSING; }
    let word=row[texel/32u];let bit=1u<<(texel%32u);
    if (word&bit)==0u { return SPATIAL_MISSING; }
    let prefix=select(0u,countOneBits(row.x),texel>=32u);
    return row.z+prefix+countOneBits(word&(bit-1u));
}
