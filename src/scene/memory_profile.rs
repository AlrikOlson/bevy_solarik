//! Explicit allocator diagnostics; absent from ordinary builds and disabled by default.
use bevy_render::renderer::RenderDevice;
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::OnceLock,
};

const MAX_SAMPLES: u32 = 64;
const MAX_ALLOCATIONS: usize = 8192;

pub(super) fn dense_reference() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SOLARIK_DIAGNOSTIC_DENSE_SCENE").is_ok_and(|v| v == "1"))
}

#[derive(Default)]
pub(super) struct MemoryProfile {
    initialized: bool,
    writer: Option<BufWriter<File>>,
    samples: u32,
}
impl MemoryProfile {
    pub(super) fn sample(
        &mut self,
        device: &RenderDevice,
        frame: Option<u32>,
        slots: usize,
        tlas: usize,
    ) {
        if !self.initialized {
            self.initialized = true;
            if let Some(path) = std::env::var_os("SOLARIK_DIAGNOSTIC_ALLOCATOR") {
                match OpenOptions::new().write(true).create_new(true).open(path) {
                    Ok(file) => {
                        let mut writer = BufWriter::new(file);
                        if writeln!(writer, "frame\tkind\tname\tbytes\taux0\taux1\taux2")
                            .and_then(|_| writer.flush())
                            .is_ok()
                        {
                            self.writer = Some(writer);
                        }
                    }
                    Err(error) => tracing::warn!("Allocator diagnostic unavailable: {error}"),
                }
            }
        }
        let Some(frame) = frame else { return };
        if !due(frame, self.samples) {
            return;
        }
        let Some(writer) = &mut self.writer else {
            return;
        };
        self.samples += 1;
        let result = (|| -> std::io::Result<()> {
            let Some(report) = device.wgpu_device().generate_allocator_report() else {
                return row(writer, frame, "unavailable", "backend", [0; 4]);
            };
            if report.allocations.len() > MAX_ALLOCATIONS || report.blocks.len() > MAX_ALLOCATIONS {
                return row(
                    writer,
                    frame,
                    "unavailable",
                    "report-limit",
                    [
                        report.allocations.len() as u64,
                        report.blocks.len() as u64,
                        0,
                        0,
                    ],
                );
            }
            row(
                writer,
                frame,
                "total",
                "prepare_begin",
                [
                    report.total_allocated_bytes,
                    report.total_reserved_bytes,
                    slots as u64,
                    tlas as u64,
                ],
            )?;
            for allocation in report.allocations {
                row(
                    writer,
                    frame,
                    "allocation",
                    &allocation.name,
                    [allocation.size, allocation.offset, 0, 0],
                )?;
            }
            for block in report.blocks {
                row(
                    writer,
                    frame,
                    "block",
                    "",
                    [
                        block.size,
                        block.allocations.start as u64,
                        block.allocations.end as u64,
                        0,
                    ],
                )?;
            }
            // An explicit terminator detects partial writes without treating them as complete snapshots.
            row(writer, frame, "end", "", [0; 4])
        })()
        .and_then(|_| writer.flush());
        if let Err(error) = result {
            tracing::warn!("Allocator diagnostic write failed: {error}");
            self.writer = None;
        }
    }
}
fn due(frame: u32, samples: u32) -> bool {
    samples < MAX_SAMPLES && frame.is_multiple_of(120)
}
fn row(
    writer: &mut impl Write,
    frame: u32,
    kind: &str,
    name: &str,
    words: [u64; 4],
) -> std::io::Result<()> {
    let name = name.replace(['\t', '\r', '\n'], " ");
    writeln!(
        writer,
        "{frame}\t{kind}\t{name}\t{}\t{}\t{}\t{}",
        words[0], words[1], words[2], words[3]
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshots_bound_work_and_keep_exact_bytes_and_frame_with_escaped_labels() {
        let frames: Vec<_> = (0..20_000).filter(|&f| due(f, f / 120)).collect();
        assert_eq!(frames.len(), MAX_SAMPLES as usize);
        assert_eq!(*frames.last().unwrap(), 7560);
        let mut bytes = Vec::new();
        row(
            &mut bytes,
            240,
            "total",
            "prepare_begin",
            [123, 4096, 8192, 4096],
        )
        .unwrap();
        row(
            &mut bytes,
            240,
            "allocation",
            "source\tmesh\n",
            [123, u64::MAX, 0, 0],
        )
        .unwrap();
        row(&mut bytes, 240, "end", "", [0; 4]).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let rows: Vec<_> = text
            .lines()
            .map(|r| r.split('\t').collect::<Vec<_>>())
            .collect();
        assert!(rows.iter().all(|r| r.len() == 7 && r[0] == "240"));
        assert_eq!(rows[0][3..], ["123", "4096", "8192", "4096"]);
        assert_eq!(rows[1][2], "source mesh ");
        assert_eq!(rows[1][4], u64::MAX.to_string());
        assert_eq!(rows[2][1], "end");
    }
}
