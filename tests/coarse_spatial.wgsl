struct Ray { origin: vec4f, direction: vec4f }
struct Result { depth: vec4f, counts: vec4u, first: vec4u, normal: vec4f }
@group(0) @binding(0) var<storage,read> grid: CoarseGrid;
@group(0) @binding(1) var<storage,read> lookup: array<u32>;
@group(0) @binding(2) var<storage,read> rows: array<vec4u>;
@group(0) @binding(6) var<storage,read> hits: array<u32>;
@group(0) @binding(3) var<storage,read> rays: array<Ray>;
@group(0) @binding(4) var<storage,read_write> output: array<Result>;
@group(0) @binding(5) var<storage,read> settings: array<u32>;
fn spatial_trace(ray: Ray) -> Result {
    var result: Result;
    var walk=coarse_walk_begin(grid,ray.origin.xyz,ray.direction.xyz,vec2f(ray.origin.w,ray.direction.w));
    loop {
        let interval=coarse_walk_next(grid,&walk);
        if interval.valid!=0u {
            let cell=lookup[interval.grid_index];
            if cell!=COARSE_MISSING {
                let z=interval.grid_index%grid.dimensions.z;
                let q=interval.grid_index/grid.dimensions.z;
                let relative=vec3i(i32(q/grid.dimensions.y),i32(q%grid.dimensions.y),i32(z));
                let address=grid.origin_resolution.xyz+relative;
                result.counts.x+=1u; result.counts.y|=settings[4u+cell];
                // A normal-dominant patch extends at most 1/8 cell width along
                // its projection axis. Probe that axial halo; keep hit ordering
                // in the destination cell, independent of the source sample.
                for(var axis=0u;axis<3u;axis+=1u) {
                    var columns=spatial_columns(address,f32(grid.origin_resolution.w),axis,
                        ray.origin.xyz,ray.direction.xyz,interval.low,interval.high);
                    loop {
                        let column=spatial_columns_next(&columns);
                        if column.valid!=0u {
                            for(var neighbor=-1;neighbor<=1;neighbor+=1) {
                                var local=relative;local[axis]+=neighbor;
                                if any(local<vec3i(0)) || any(local>=vec3i(grid.dimensions.xyz)) { continue; }
                                let index=u32((local.x*i32(grid.dimensions.y)+local.y)*i32(grid.dimensions.z)+local.z);
                                let source=lookup[index];if source==COARSE_MISSING { continue; }
                                for(var side=0u;side<2u;side+=1u) {
                                    let sample_index=spatial_sample_index(rows[source*6u+axis*2u+side],column.grid_index);
                                    var word=0u;
                                    if sample_index!=SPATIAL_MISSING { word=hits[sample_index]; }
                                    let hit=spatial_intersect(word,grid.origin_resolution.xyz+local,
                                        f32(grid.origin_resolution.w),axis,column.grid_index,settings[2],
                                        ray.origin.xyz,ray.direction.xyz,column.low,column.high);
                                    if hit.valid!=0u && (result.depth.x==0.0 || hit.t<result.depth.y) {
                                        result.depth=vec4f(1.0,hit.t,0.0,0.0);
                                        result.first=vec4u(hit.part+1u,source,column.grid_index,sample_index);
                                        result.normal=vec4f(hit.normal,0.0);
                                    }
                                }
                            }
                        }
                        if columns.running==0u { break; }
                    }
                    result.counts.z+=columns.visited;result.counts.w|=columns.overflow;
                }
            }
        }
        if result.depth.x>0.0 || walk.running==0u || result.counts.w!=0u { break; }
    }
    result.counts.w|=walk.overflow; return result;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x<arrayLength(&rays) { output[id.x]=spatial_trace(rays[id.x]); }
}
@compute @workgroup_size(64)
fn quantization(@builtin(global_invocation_id) id: vec3u) {
    if id.x>=arrayLength(&rays) { return; }
    let ray=rays[id.x]; var result: Result;
    let word=spatial_pack(ray.direction.w,ray.direction.xyz,id.x%4u);
    result.depth.x=spatial_depth(word);
    result.first=vec4u(word,spatial_part(word),0u,0u);
    result.normal=vec4f(spatial_normal(word),0.0); output[id.x]=result;
}
