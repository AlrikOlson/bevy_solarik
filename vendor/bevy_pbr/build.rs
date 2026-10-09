//! Embed the actual immutable compiler recipe for derived meshlet invalidation.
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn append(recipe: &mut Vec<u8>, name: &str, bytes: &[u8]) {
    recipe.extend_from_slice(&(name.len() as u64).to_le_bytes());
    recipe.extend_from_slice(name.as_bytes());
    recipe.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    recipe.extend_from_slice(bytes);
}

fn tree(recipe: &mut Vec<u8>, root: &Path, path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .expect("compiler recipe directory")
            .map(|e| e.expect("compiler recipe entry").path())
            .collect();
        entries.sort();
        for entry in entries {
            tree(recipe, root, &entry);
        }
    } else {
        append(
            recipe,
            &path
                .strip_prefix(root)
                .expect("recipe root")
                .to_string_lossy(),
            &fs::read(path).expect("compiler recipe source"),
        );
    }
}

fn version(command: &str, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new(command).args(args).output().ok()?;
    let mut bytes = output.stdout;
    bytes.extend(output.stderr);
    (!bytes.is_empty()).then_some(bytes)
}

fn compiler(recipe: &mut Vec<u8>, meshopt: &Path) -> Option<()> {
    // Mirror the owned meshopt build's compiler selection. In particular MSVC
    // leaves cpp(false), so using CXX or searching PATH for cl is incorrect.
    let target = env::var("TARGET").ok()?;
    let mut build = cc::Build::new();
    build.cargo_metadata(false).include(meshopt.join("src"));
    if target.contains("darwin") {
        build
            .flag("-std=c++11")
            .cpp_link_stdlib("c++")
            .cpp_set_stdlib("c++")
            .cpp(true);
    } else if target.contains("linux") || target.contains("windows-gnu") {
        build.flag("-std=c++11").cpp_link_stdlib("stdc++").cpp(true);
    }
    if env::var("DEBUG").ok()?.as_str() != "true" {
        build.define("NDEBUG", None);
    }
    if target.starts_with("wasm32") {
        build.flag("-isystem").flag("include_wasm32");
    }
    let tool = build.try_get_compiler().ok()?;
    let path = tool.path().canonicalize().ok()?;
    append(recipe, "cpp-path", path.to_string_lossy().as_bytes());
    append(recipe, "cpp-args", format!("{:?}", tool.args()).as_bytes());
    append(recipe, "cpp-env", format!("{:?}", tool.env()).as_bytes());
    fingerprint_binary(recipe, &path)?;
    // These perform the actual MSVC C++ frontend/backend work. A toolchain
    // update must invalidate data even when the driver executable is unchanged.
    if tool.is_like_msvc() && !tool.is_like_clang_cl() {
        let directory = path.parent()?;
        for library in ["c1xx.dll", "c2.dll"] {
            fingerprint_binary(recipe, &directory.join(library))?;
        }
    }
    let mut command = tool.to_command();
    command.arg(if tool.is_like_msvc() && !tool.is_like_clang_cl() {
        "/Bv"
    } else {
        "--version"
    });
    if let Ok(output) = command.output() {
        append(recipe, "cpp-version-stdout", &output.stdout);
        append(recipe, "cpp-version-stderr", &output.stderr);
    }
    Some(())
}

fn fingerprint_binary(recipe: &mut Vec<u8>, path: &Path) -> Option<()> {
    println!("cargo:rerun-if-changed={}", path.display());
    let bytes = fs::read(path).ok()?;
    append(
        recipe,
        &path.to_string_lossy(),
        blake3::hash(&bytes).as_bytes(),
    );
    Some(())
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("package directory"));
    let mut recipe = Vec::new();
    let mut known = false;
    if env::var_os("CARGO_FEATURE_MESHLET_PROCESSOR").is_some() {
        for source in [
            "asset.rs",
            "from_mesh.rs",
            "lod_area.rs",
            "lod_attributes.rs",
            "lod_spatial.rs",
            "tangent_data.rs",
            "tangent_palette.rs",
            "derived_cache.rs",
            "derived_cache_retention.rs",
        ] {
            tree(&mut recipe, &root, &root.join("src/meshlet").join(source));
        }
        tree(&mut recipe, &root, &root.join("Cargo.toml"));
        tree(&mut recipe, &root, &root.join("build.rs"));
        // Include every owned meshoptimizer implementation and wrapper input.
        let meshopt = root.parent().expect("vendor directory").join("meshopt");
        for path in ["src", "gen", "vendor/src", "Cargo.toml", "build.rs"] {
            tree(&mut recipe, &meshopt, &meshopt.join(path));
        }
        let lock = root
            .parent()
            .and_then(Path::parent)
            .expect("renderer directory")
            .join("Cargo.lock");
        println!("cargo:rerun-if-changed={}", lock.display());
        append(
            &mut recipe,
            "dependency-lock",
            &fs::read(lock).expect("renderer dependency lock"),
        );
        for variable in [
            "TARGET",
            "PROFILE",
            "OPT_LEVEL",
            "DEBUG",
            "CARGO_CFG_TARGET_FEATURE",
            "CARGO_ENCODED_RUSTFLAGS",
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "VCToolsVersion",
            "WindowsSDKVersion",
        ] {
            println!("cargo:rerun-if-env-changed={variable}");
            append(
                &mut recipe,
                variable,
                env::var(variable).unwrap_or_default().as_bytes(),
            );
        }
        // cc supports target-specific compiler/flag variables as well.
        let mut flags: Vec<_> = env::vars()
            .filter(|(name, _)| {
                [
                    "CC_",
                    "CXX_",
                    "CFLAGS_",
                    "CXXFLAGS_",
                    "HOST_CC",
                    "HOST_CXX",
                    "HOST_CFLAGS",
                    "HOST_CXXFLAGS",
                ]
                .iter()
                .any(|prefix| name.starts_with(prefix))
                    || (name.starts_with("CARGO_TARGET_") && name.ends_with("_RUSTFLAGS"))
                    || matches!(name.as_str(), "RUSTC_WRAPPER" | "RUSTC_WORKSPACE_WRAPPER")
            })
            .collect();
        flags.sort();
        for (name, value) in flags {
            println!("cargo:rerun-if-env-changed={name}");
            append(&mut recipe, &name, value.as_bytes());
        }
        let rust = env::var("RUSTC")
            .ok()
            .and_then(|rust| version(&rust, &["--version", "--verbose"]));
        if let (Some(rust), Some(())) = (rust, compiler(&mut recipe, &meshopt)) {
            append(&mut recipe, "rust-compiler", &rust);
            known = true;
        }
    }
    println!(
        "cargo:rustc-env=MESHLET_CACHE_COMPILER_KNOWN={}",
        u8::from(known)
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("build output"));
    fs::write(output.join("meshlet-compiler-recipe.bin"), recipe).expect("compiler recipe output");
}
