#!/usr/bin/env bash
# Regenerate docs/upstream-0.19.1.diff: everything this crate changes against
# the crates.io bevy_solari 0.19.1 sources in the local cargo registry. The
# crate rename (bevy_solari:: -> bevy_solarik::, Solari* -> Solarik*) is in
# there too, so read it as provenance, not as a patch to apply upstream.
#   tools/upstream_diff.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
registry="$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/bevy_solari-0.19.1 | head -n1)"
[ -d "$registry" ] || { echo "bevy_solari-0.19.1 not in the cargo registry (cargo fetch first)" >&2; exit 1; }
out="$here/docs/upstream-0.19.1.diff"
mkdir -p "$here/docs"
cd "$(dirname "$registry")"
# diff exits 1 when the trees differ, which is the point. Packaging files
# (manifest, README, this script, the diff itself) are left out so the diff
# is the code.
diff -ruN --strip-trailing-cr \
  -x .cargo-ok -x .cargo_vcs_info.json -x .cargo -x Cargo.lock -x Cargo.toml -x Cargo.toml.orig \
  -x README.md -x docs -x tools -x target \
  bevy_solari-0.19.1 "$here" > "$out" || [ $? -eq 1 ]
# Relative paths in the headers (Git bash spells the crate dir /d/..., Windows
# tools D:/...).
win="$(cygpath -m "$here" 2>/dev/null || echo "$here")"
sed -i "s#$win#bevy_solarik#g; s#$here#bevy_solarik#g" "$out"
echo "wrote ${out#$here/} ($(wc -l < "$out") lines)"
