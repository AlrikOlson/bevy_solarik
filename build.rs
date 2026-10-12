//! Native link requirements for optional DLSS consumers.

#[expect(
    clippy::print_stdout,
    reason = "Cargo build scripts communicate native link requirements on stdout."
)]
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var_os("CARGO_FEATURE_DLSS").is_some()
    {
        // NGX's registry imports are not declared by dlss_wgpu 6. Standalone
        // consumers must not rely on a window plugin linking them incidentally.
        println!("cargo:rustc-link-lib=advapi32");
    }
}
