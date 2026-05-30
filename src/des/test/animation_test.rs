//! Port of src/des/test/animation-test.ts
//
// The `animation` subsystem (`frame_recorder`, `types`, `html_player`) is now
// ported. This exercises the FrameRecorder lifecycle end to end: open a JSONL
// sink, record frames, and read them back via `finish()`.

#[cfg(test)]
mod tests {
    use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
    use crate::des::animation::types::FrameParts;

    fn opts(frames_path: &str) -> FrameRecorderOpts {
        FrameRecorderOpts {
            frames_path: frames_path.to_string(),
            html_path: None,
            width: 320.0,
            height: 240.0,
            fps: None,
            title: None,
            subtitle: None,
            background: None,
            live_tick_line: None,
            record_every_ticks: None,
            visual_blocks: None,
        }
    }

    #[test]
    fn frame_recorder_records_and_reads_back_frames() {
        let path = std::env::temp_dir()
            .join(format!("des_anim_test_{}.jsonl", std::process::id()));
        let path_str = path.to_string_lossy().into_owned();

        let mut rec = FrameRecorder::new(opts(&path_str)).expect("create recorder");
        assert_eq!(rec.get_frame_count(), 0);

        rec.frame(0.0, 0.0, || FrameParts::new(vec![]));
        rec.frame(1.0, 1.0, || FrameParts::with_caption(vec![], "tick 1"));
        assert_eq!(rec.get_frame_count(), 2);

        let anim = rec.finish().expect("finish recorder");
        assert_eq!(anim.frames.len(), 2, "read-back frame count mismatch");
        assert!(path.exists(), "frames file was not written");

        let _ = std::fs::remove_file(&path);
    }
}
