//! Select production WESL fragments for fixtures that provide independent bindings and oracles.
extern crate alloc;

pub fn wgsl(source: alloc::borrow::Cow<'_, str>) -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Wgsl(select(&source, &[]).into())
}

pub fn select(source: &str, defines: &[&str]) -> String {
    let mut module: wesl::syntax::TranslationUnit = source.parse().expect("production WESL syntax");
    let mut features = wesl::Features::new();
    features.default = wesl::Feature::Disable;
    for define in defines {
        features.set(define, true);
    }
    wesl::pass::condcomp(&mut module, &features).expect("production WESL variant");
    // The fixture supplies these dependencies explicitly; full graph tests verify their linking.
    module.imports.clear();
    if source.contains("import bevy_solarik::cloud_field") {
        alloc::format!("{}\n{}", include_str!("../../src/cloud_field.wesl"), module)
    } else {
        module.to_string()
    }
}
