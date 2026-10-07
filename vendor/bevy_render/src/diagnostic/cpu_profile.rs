//! Opt-in, bounded CPU stage samples for native renderer profiling.
use core::sync::atomic::{AtomicU64, Ordering};
use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};

const MAX_PENDING_SAMPLES: usize = 4096;
static DROPPED: AtomicU64 = AtomicU64::new(0);
static ENABLED: OnceLock<bool> = OnceLock::new();
static SAMPLES: Mutex<Vec<CpuProfileSample>> = Mutex::new(Vec::new());

/// One active CPU duration or a scene counter; nested durations must not be summed.
#[derive(Debug)]
pub struct CpuProfileSample {
    pub name: &'static str,
    pub value: f64,
    pub unit: &'static str,
}

/// Enable before process startup with `BEVY_RENDER_PROFILE=1`.
pub fn profile_enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("BEVY_RENDER_PROFILE").is_some())
}

/// Times active CPU work, excluding the GPU's execution time.
pub fn profile_scope(name: &'static str) -> Option<CpuProfileGuard> {
    profile_enabled().then(|| CpuProfileGuard {
        name,
        start: Instant::now(),
    })
}

pub struct CpuProfileGuard {
    name: &'static str,
    start: Instant,
}

impl Drop for CpuProfileGuard {
    fn drop(&mut self) {
        profile_value(self.name, self.start.elapsed().as_secs_f64() * 1000.0, "ms");
    }
}

/// Record counts/bytes separately from durations.
pub fn profile_value(name: &'static str, value: f64, unit: &'static str) {
    if !profile_enabled() || !value.is_finite() {
        return;
    }
    let mut samples = SAMPLES.lock().expect("CPU profile samples");
    if !append_sample(&mut samples, CpuProfileSample { name, value, unit }) {
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
                },
            );
        }
        assert_eq!(samples.len(), MAX_PENDING_SAMPLES);
    }
}
