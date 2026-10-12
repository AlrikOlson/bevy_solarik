//! Link the same physical WESL modules and compiler used by Bevy's pipeline cache.
use bevy_asset::{AssetId, Assets};
use bevy_shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};

pub struct Graph {
    cache: ShaderCache<String, ()>,
    ids: BTreeMap<String, AssetId<Shader>>,
    assets: Assets<Shader>,
}
impl Graph {
    pub fn load() -> Self {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--locked", "--offline"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("locked dependency metadata");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let mut graph = Self {
            cache: ShaderCache::new((), |_, source, _| match source {
                ShaderCacheSource::Wgsl(source) => Ok(source),
                _ => panic!("WESL graph must produce WGSL"),
            }),
            ids: BTreeMap::new(),
            assets: Assets::default(),
        };
        for package in metadata["packages"].as_array().unwrap() {
            let name = package["name"].as_str().unwrap();
            if !(name.starts_with("bevy_") || name == "space-sim") {
                continue;
            }
            let manifest = Path::new(package["manifest_path"].as_str().unwrap());
            let root = manifest.parent().unwrap().join("src");
            let mut files = Vec::new();
            collect(&root, &mut files);
            files.sort();
            for file in files {
                let relative = file
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let package_name = name.replace('-', "_");
                let module = format!(
                    "{}::{}",
                    package_name,
                    relative.trim_end_matches(".wesl").replace('/', "::")
                );
                let path = format!("embedded://{package_name}/{relative}");
                let shader = Shader::from_wesl(std::fs::read_to_string(&file).unwrap(), path);
                let handle = graph.assets.add(shader.clone());
                graph.ids.insert(module.clone(), handle.id());
                graph.cache.set_shader(handle.id(), shader);
            }
        }
        graph
    }

    pub fn compile(&mut self, module: &str, defs: &[ShaderDefVal]) -> Result<String, String> {
        let id = *self
            .ids
            .get(module)
            .ok_or_else(|| format!("unregistered root {module}"))?;
        self.cache
            .get(0, id, defs)
            .map(|source| (*source).clone())
            .map_err(|error| error.to_string())
    }

    pub fn validate(&mut self, module: &str, defs: &[ShaderDefVal]) {
        let source = self
            .compile(module, defs)
            .unwrap_or_else(|e| panic!("{module}: {e}"));
        let parsed = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{module}: {}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap_or_else(|e| panic!("{module}: {}", e.emit_to_string(&source)));
    }
}
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if !path.exists() {
        return;
    }
    for entry in std::fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "wesl") {
            files.push(path);
        }
    }
}
