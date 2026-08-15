---
name: ffmpeg/ffprobe clean LD_LIBRARY_PATH
description: ffprobe/ffmpeg fail with MOUNT_2_40 not found when server.ts prepends system lib paths to LD_LIBRARY_PATH for canvas.node.
---

## The Rule
Always pass `{ env: makeCleanEnv() }` when calling `execSync`/`spawn` for ffprobe or ffmpeg. The `makeCleanEnv()` helper (in `server/pipeline.ts`) strips `/lib/x86_64-linux-gnu` and `/usr/lib/x86_64-linux-gnu` from `LD_LIBRARY_PATH` so Nix binaries resolve their own libs via rpath, not the older system libs.

**Why:** server.ts prepends system lib paths so canvas.node workers find `libuuid.so.1`. But Nix-built ffprobe/ffmpeg then pick up the system `libmount.so.1`, which is older and lacks the `MOUNT_2_40` versioned symbol that Nix glib (`libgio-2.0.so.0`) requires — causing an immediate crash.

**How to apply:** Any new `spawn`/`execSync` for a Nix binary in `server/pipeline.ts` should use `{ env: makeCleanEnv() }`.

**Note:** canvas.node itself does NOT need LD_LIBRARY_PATH because the libuuid symlink at `node_modules/canvas/build/Release/libuuid.so.1` covers it via `$ORIGIN` RUNPATH. Recreate the symlink after `npm install` if canvas fails: `ln -sf /lib/x86_64-linux-gnu/libuuid.so.1 node_modules/canvas/build/Release/libuuid.so.1`.
