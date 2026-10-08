// Experimental render placement contract, scalar-word input.
// Bindings and page/instance identity are supplied by the scene consumer.
struct PlacementPage {
    origin_height: vec4<f32>,
    extent_height: vec4<f32>,
}
struct DecodedPlacement {
    root: vec3<f32>,
    height: f32,
    rotation: vec4<f32>,
    prototype: u32,
}

fn placement_quaternion(encoded: vec3<u32>, largest: u32, maximum: f32) -> vec4<f32> {
    let small = vec3<f32>(encoded) / maximum * 1.4142135623730951 - 0.7071067811865476;
    let omitted = sqrt(max(0.0, 1.0 - dot(small, small)));
    var result: vec4<f32>;
    var j = 0u;
    for (var i = 0u; i < 4u; i += 1u) {
        if i == largest {
            result[i] = omitted;
        } else {
            result[i] = small[j];
            j += 1u;
        }
    }
    return result;
}

fn placement_decode(
    word0: u32, word1: u32, height: u32, prototype: u32,
    rotation: vec4<f32>, maximum: f32, page: PlacementPage,
) -> DecodedPlacement {
    let position = vec3<u32>(word0 & 65535u, word0 >> 16u, word1 & 65535u);
    let local = vec3<f32>(position) / 65535.0 * page.extent_height.xyz;
    return DecodedPlacement(
        page.origin_height.xyz + local,
        f32(height) / maximum * page.extent_height.w + page.origin_height.w,
        rotation, prototype,
    );
}

fn placement_decode12(word0: u32, word1: u32, word2: u32, page: PlacementPage) -> DecodedPlacement {
    let encoded = vec3<u32>(word2 & 1023u, (word2 >> 10u) & 1023u, (word2 >> 20u) & 1023u);
    let rotation = placement_quaternion(encoded, word2 >> 30u, 1023.0);
    return placement_decode(word0, word1, (word1 >> 16u) & 1023u, word1 >> 26u, rotation, 1023.0, page);
}

fn placement_decode16(
    word0: u32, word1: u32, word2: u32, word3: u32, page: PlacementPage,
) -> DecodedPlacement {
    let encoded = vec3<u32>(word2 & 65535u, word2 >> 16u, word3 & 65535u);
    let rotation = placement_quaternion(encoded, word3 >> 30u, 65535.0);
    return placement_decode(word0, word1, word1 >> 16u, (word3 >> 16u) & 16383u, rotation, 65535.0, page);
}

fn placement_transform(placement: DecodedPlacement, point: vec3<f32>) -> vec3<f32> {
    let q = placement.rotation;
    let rotated = point + 2.0 * cross(q.xyz, cross(q.xyz, point) + q.w * point);
    return placement.root + placement.height * rotated;
}
