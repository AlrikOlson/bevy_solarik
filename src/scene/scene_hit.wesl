import bevy_solarik::filtered_canopy_scene::ForestSceneEvent;

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
    coarse: ForestSceneEvent,
}
fn scene_miss() -> SceneHit {
    var hit: SceneHit;
    return hit;
}
fn scene_triangle(t: f32, slot: u32, primitive: u32, barycentrics: vec2f, front_face: bool) -> SceneHit {
    var hit:SceneHit;
    hit.kind=SCENE_HIT_TRIANGLE;hit.t=t;
    hit.triangle=SceneTriangleHit(slot, primitive, barycentrics, front_face);
    return hit;
}
fn scene_coarse(event:ForestSceneEvent)->SceneHit {
    var hit:SceneHit;hit.kind=SCENE_HIT_COARSE;hit.t=event.event.position_distance.w;
    hit.coarse=event;return hit;
}
fn scene_hit_is_coarse(hit:SceneHit)->bool {return hit.kind==SCENE_HIT_COARSE && hit.coarse.event.normal_hit.w>0.0;}
fn scene_hit_is_miss(hit: SceneHit) -> bool { return hit.kind == SCENE_HIT_MISS; }
fn scene_hit_is_triangle(hit: SceneHit) -> bool { return hit.kind == SCENE_HIT_TRIANGLE; }
