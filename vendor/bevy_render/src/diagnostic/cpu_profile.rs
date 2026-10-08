//! Opt-in, bounded CPU stage samples for native renderer profiling.
use alloc::rc::Rc;
use core::{
    cell::Cell,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};
use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};

const MAX_PENDING_SAMPLES: usize = 4096;
static DROPPED: AtomicU64 = AtomicU64::new(0);
static ENABLED: OnceLock<bool> = OnceLock::new();
static SAMPLES: Mutex<Vec<CpuProfileSample>> = Mutex::new(Vec::new());
thread_local! { static SOURCE_FRAME: Cell<Option<u32>> = const { Cell::new(None) }; }

/// One active CPU duration or a scene counter; nested durations must not be summed.
#[derive(Debug)]
pub struct CpuProfileSample {
    pub name: &'static str,
    pub value: f64,
    pub unit: &'static str,
    /// Original source, independent of worker/main delivery time.
    pub source_render_frame: Option<u32>,
}

/// Enable before process startup with `BEVY_RENDER_PROFILE=1`.
pub fn profile_enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("BEVY_RENDER_PROFILE").is_some())
}

/// Tag nested counters and scopes on this thread without changing other workers.
pub fn profile_source_frame(frame: u32) -> Option<CpuProfileSourceGuard> {
    profile_enabled().then(|| CpuProfileSourceGuard::enter(frame))
}

/// A source context restores its enclosing context and cannot move to another thread.
pub struct CpuProfileSourceGuard {
    previous: Option<u32>,
    thread: PhantomData<Rc<()>>,
}
impl CpuProfileSourceGuard {
    fn enter(frame: u32) -> Self {
        Self {
            previous: SOURCE_FRAME.with(|f| f.replace(Some(frame))),
            thread: PhantomData,
        }
    }
}
impl Drop for CpuProfileSourceGuard {
    fn drop(&mut self) {
        SOURCE_FRAME.with(|f| f.set(self.previous));
    }
}

/// Times active CPU work, excluding the GPU's execution time.
pub fn profile_scope(name: &'static str) -> Option<CpuProfileGuard> {
    profile_enabled().then(|| CpuProfileGuard {
        name,
        start: Instant::now(),
        source_render_frame: SOURCE_FRAME.with(Cell::get),
    })
}
pub struct CpuProfileGuard {
    name: &'static str,
    start: Instant,
    source_render_frame: Option<u32>,
}
impl Drop for CpuProfileGuard {
    fn drop(&mut self) {
        record_sample(
            self.name,
            self.start.elapsed().as_secs_f64() * 1000.0,
            "ms",
            self.source_render_frame,
        );
    }
}

/// Record counts/bytes separately from durations.
pub fn profile_value(name: &'static str, value: f64, unit: &'static str) {
    if profile_enabled() {
        record_sample(name, value, unit, SOURCE_FRAME.with(Cell::get));
    }
}
fn record_sample(
    name: &'static str,
    value: f64,
    unit: &'static str,
    source_render_frame: Option<u32>,
) {
    if !value.is_finite() {
        return;
    }
    let mut samples = SAMPLES.lock().expect("CPU profile samples");
    if !append_sample(
        &mut samples,
        CpuProfileSample {
            name,
            value,
            unit,
            source_render_frame,
        },
    ) {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}
fn append_sample(samples: &mut Vec<CpuProfileSample>, sample: CpuProfileSample) -> bool {
    if samples.len() >= MAX_PENDING_SAMPLES {
        return false;
    }
    samples.push(sample);
    true
}

/// Drain once per main frame. Samples can come from worker and render threads.
pub fn take_cpu_profile() -> Vec<CpuProfileSample> {
    let mut samples = core::mem::take(&mut *SAMPLES.lock().expect("CPU profile samples"));
    let dropped = DROPPED.swap(0, Ordering::Relaxed);
    if dropped != 0 {
        samples.push(CpuProfileSample {
            name: "profile.dropped",
            value: dropped as f64,
            unit: "count",
            source_render_frame: None,
        });
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_consumer_cannot_grow_samples_without_bound() {
        let mut samples = Vec::new();
        for _ in 0..MAX_PENDING_SAMPLES + 1 {
            append_sample(
                &mut samples,
                CpuProfileSample {
                    name: "test",
                    value: 2.0,
                    unit: "ms",
                    source_render_frame: None,
                },
            );
        }
        assert_eq!(samples.len(), MAX_PENDING_SAMPLES);
    }
    #[test]
    fn source_context_restores_nesting_and_duration_keeps_its_frame_across_workers() {
        let prior = SOURCE_FRAME.with(Cell::get);
        let outer = CpuProfileSourceGuard::enter(17);
        let inner = CpuProfileSourceGuard::enter(12);
        record_sample(
            "source.test.inner",
            42.0,
            "count",
            SOURCE_FRAME.with(Cell::get),
        );
        drop(inner);
        let duration = CpuProfileGuard {
            name: "source.test.duration",
            start: Instant::now(),
            source_render_frame: SOURCE_FRAME.with(Cell::get),
        };
        std::thread::spawn(move || drop(duration)).join().unwrap();
        drop(outer);
        assert_eq!(SOURCE_FRAME.with(Cell::get), prior);
        let samples: Vec<_> = take_cpu_profile()
            .into_iter()
            .filter(|s| s.name.starts_with("source.test."))
            .collect();
        assert_eq!(
            samples
                .iter()
                .map(|s| s.source_render_frame)
                .collect::<Vec<_>>(),
            [Some(12), Some(17)]
        );
    }
}
