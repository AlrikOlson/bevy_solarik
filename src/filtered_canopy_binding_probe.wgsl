// Bound every read and loop: reproduce stale descriptor ranges without hanging the GPU.
@group(0) @binding(0) var<storage, read> scene: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> result: vec4u;
@compute @workgroup_size(1)
fn probe() {
    let length = arrayLength(&scene);
    let header = scene[0];
    var errors = 0u;
    if header.y + header.x * 2u > length || header.z + header.w * 14u > length {
        errors |= 1u;
    }
    for (var node = 0u; node < min(header.x, 4096u); node += 1u) {
        let address = header.y + node * 2u;
        if address + 1u >= length { errors |= 2u; break; }
        let low = scene[address];
        let high = scene[address + 1u];
        if low.w <= node || low.w > header.x { errors |= 4u; }
        if high.w != 0xffffffffu && high.w >= header.w { errors |= 8u; }
    }
    result = vec4u(length, header.x, header.w, errors);
}
