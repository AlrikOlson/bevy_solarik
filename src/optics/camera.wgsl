struct Parameters { meter:vec4<f32>, lens:vec4<f32>, bounds:vec4<f32>, frame:vec4<f32> }
@group(0) @binding(0) var source:texture_2d<f32>;
#ifdef METER
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var<storage,read_write> state:array<vec4<f32>,3>;
@compute @workgroup_size(1)
fn meter() {
    var histogram:array<u32,256>;
    let dim=textureDimensions(source);
    var count=0u;
    // Stratified central50% spot meter; positive scene luminance only, 4096 samples.
    for(var y=0u;y<64u;y++) { for(var x=0u;x<64u;x++) {
        let uv=vec2<f32>(0.25)+0.5*(vec2<f32>(f32(x),f32(y))+0.5)/64.0;
        let rgb=textureLoad(source,vec2<i32>(uv*vec2<f32>(dim)),0).rgb;
        let lum=dot(rgb,vec3<f32>(0.2126,0.7152,0.0722))/p.frame.x;
        if lum>0.000001 && lum<1e10 {
            let bin=u32(clamp((log2(lum)+20.0)*256.0/60.0,0.0,255.0));
            histogram[bin]+=1u; count+=1u;
        }
    }}
    var cumulative=0.0; var weighted=0.0; var included=0.0;
    let low=f32(count)*0.75; let high=f32(count)*0.95;
    for(var i=0u;i<256u;i++) {
        let next=cumulative+f32(histogram[i]);
        let n=max(0.0,min(next,high)-max(cumulative,low));
        weighted+=n*(-20.0+(f32(i)+0.5)*60.0/256.0); included+=n; cumulative=next;
    }
    var desired=p.meter.y; var lum=0.0;
    if included>0.0 {lum=exp2(weighted/included);}
    // A resolved source directly under the aim must not fall between sparse
    // histogram samples. This is a declared centre-spot photographic policy.
    var spot=0.0;
    for(var y=-2;y<2;y++) {for(var x=-2;x<2;x++) {
        let pixel=clamp(vec2<i32>(dim/2u)+vec2(x,y),vec2(0),vec2<i32>(dim)-1);
        let value=dot(textureLoad(source,pixel,0).rgb,vec3(0.2126,0.7152,0.0722))/p.frame.x;
        if value>0.000001 && value<1e10 {spot+=value/16.0;}
    }}
    lum=max(lum,spot);
    if lum>0.000001 && lum<1e10 {desired=clamp(log2(8.0*lum)-p.meter.z,p.bounds.x,p.bounds.y);}
    var ev=state[0].x;
    if p.frame.y>0.0 {ev=p.meter.y;}
    if p.meter.x<0.5 {ev=p.meter.y;desired=ev;}
    else {
        let delta=desired-ev;
        let rate=select(p.lens.z,p.lens.w,delta>0.0);
        // Exact rate-limited relaxation, tau=1s; matches optics::adapt_ev.
        let error=abs(delta);
        let linear_time=min(max((error-rate)/rate,0.0),p.meter.w);
        let linear=rate*linear_time;
        let relaxed=(error-linear)*(1.0-exp(-(p.meter.w-linear_time)));
        ev+=sign(delta)*(linear+relaxed);
    }
    state[0]=vec4<f32>(ev,desired,lum,f32(count));
    let centre=textureLoad(source,vec2<i32>(dim/2u),0).rgb;
    state[1]=vec4<f32>(p.meter.x,p.meter.y,dot(centre,vec3(0.2126,0.7152,0.0722)),p.frame.x);
}
#else
#ifdef HORIZONTAL
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var destination:texture_storage_2d<rgba16float,write>;
#else
@group(0) @binding(1) var blurred:texture_2d<f32>;
@group(0) @binding(2) var<uniform> p:Parameters;
@group(0) @binding(3) var<storage,read_write> state:array<vec4<f32>,3>;
@group(0) @binding(4) var destination:texture_storage_2d<rgba16float,write>;
#endif
fn reflected(v:i32,n:i32)->i32 {let q=((v%(2*n))+2*n)%(2*n);return select(q,2*n-1-q,q>=n);}
@compute @workgroup_size(8,8)
fn scatter(@builtin(global_invocation_id) id:vec3<u32>) {
    let dim=textureDimensions(source);if any(id.xy>=dim) {return;}
    let pos=vec2<i32>(id.xy);let sigma=p.lens.x;
    let radius=i32(ceil(4.0*sigma));
    var total=0.0;var sum=vec3<f32>(0.0);
    for(var i=-radius;i<=radius;i++) {
        let w=exp(-0.5*pow(f32(i)/max(sigma,0.1),2.0));
#ifdef HORIZONTAL
        let q=vec2<i32>(reflected(pos.x+i,i32(dim.x)),pos.y);
        sum+=w*textureLoad(source,q,0).rgb;
#else
        let q=vec2<i32>(pos.x,reflected(pos.y+i,i32(dim.y)));
        sum+=w*textureLoad(blurred,q,0).rgb;
#endif
        total+=w;
    }
    var rgb=sum/total;
#ifndef HORIZONTAL
    rgb=mix(textureLoad(source,pos,0).rgb,rgb,p.lens.y);
    rgb*=exp2(p.meter.y-state[0].x);
    // Representable sensor-storage ceiling after exposure, never infinity.
    rgb=clamp(rgb,vec3(0.0),vec3(65000.0));
    if all(id.xy==dim/2u) {state[2]=vec4(rgb,1.0);}
#endif
    textureStore(destination,pos,vec4<f32>(rgb,textureLoad(source,pos,0).a));
}
#endif
