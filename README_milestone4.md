# Milestone 4 Results
- Built a standalone Node.js server using `child_process.spawn` to run `ffmpeg` directly (piping JPEG frames via stdout).
- Used `worker_threads` to spawn a worker per CPU core (`os.cpus().length`, which is 2 cores in this environment).
- Shared the exact same `src/shared/fingerprint.ts` logic as the browser without modification.
- Evaluated on a 2-minute (3000 frames) MP4 test video.
- Performance: Processed 3000 frames in **7.4s** (Average FPS: **407.9 FPS**) on 2 cores.

Ready for Milestone 5!
