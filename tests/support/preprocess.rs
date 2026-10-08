//! Select explicit production shader variants before passing them to plain WGSL.
pub fn select(source: &str, defines: &[&str]) -> String {
    let mut active = vec![true];
    let mut output = String::new();
    for line in source.lines() {
        let directive = line.trim();
        if let Some(name) = directive.strip_prefix("#ifdef ") {
            active.push(*active.last().expect("parent") && defines.contains(&name));
        } else if let Some(name) = directive.strip_prefix("#ifndef ") {
            active.push(*active.last().expect("parent") && !defines.contains(&name));
        } else if directive == "#else" {
            let previous = active.pop().expect("conditional");
            active.push(*active.last().expect("parent") && !previous);
        } else if directive == "#endif" {
            assert!(active.len() > 1, "unbalanced shader conditional");
            active.pop();
        } else if *active.last().expect("state")
            && !directive.starts_with('#')
            && !directive.starts_with("enable ")
        {
            output.push_str(line);
            output.push('\n');
        }
    }
    assert_eq!(active, [true], "unbalanced shader conditionals");
    output
}
