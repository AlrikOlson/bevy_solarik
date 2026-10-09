#define_import_path bevy_solarik::scene_hit
// Renderer-owned query boundary. Triangle payload is meaningful only for its tag.
// A future coarse producer must supply a resolved event, never a box entry.
const SCENE_HIT_MISS: u32 = 0u;
const SCENE_HIT_TRIANGLE: u32 = 1u;
const SCENE_HIT_COARSE: u32 = 2u;
const SCENE_HIT_INVALID: u32 = 0xffffffffu;
struct SceneTriangleHit {
    slot: u32,
    primitive: u32,
    barycentrics: vec2f,
    front_face: bool,
}
struct SceneHit {
    kind: u32,
    t: f32,
    triangle: SceneTriangleHit,
}
fn scene_miss() -> SceneHit {
    var hit: SceneHit;
    return hit;
}
fn scene_triangle(t: f32, slot: u32, primitive: u32, barycentrics: vec2f, front_face: bool) -> SceneHit {
    return SceneHit(SCENE_HIT_TRIANGLE, t, SceneTriangleHit(slot, primitive, barycentrics, front_face));
}
fn scene_hit_is_miss(hit: SceneHit) -> bool { return hit.kind == SCENE_HIT_MISS; }
fn scene_hit_is_triangle(hit: SceneHit) -> bool { return hit.kind == SCENE_HIT_TRIANGLE; }
