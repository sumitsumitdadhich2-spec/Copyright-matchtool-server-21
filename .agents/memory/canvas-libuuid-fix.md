---
name: Canvas native addon libuuid fix
description: canvas.node (node-canvas) fails to load with "libuuid.so.1: cannot open shared object file" on this NixOS Replit host.
---

## The Rule
Create a symlink `node_modules/canvas/build/Release/libuuid.so.1 → /lib/x86_64-linux-gnu/libuuid.so.1` to fix canvas loading.

**Why:** canvas.node has `RUNPATH=$ORIGIN` — it searches its own directory for .so files. libuuid.so.1 is NOT bundled (unlike libcairo, libpango, etc.) but IS available at /lib/x86_64-linux-gnu. LD_LIBRARY_PATH is empty by default in NixOS and cannot be set in the shell startup command without breaking bash (vdso conflict). The symlink approach works because $ORIGIN is checked before LD_LIBRARY_PATH.

**How to apply:** After `npm install` or `npm rebuild`, run:
```
ln -sf /lib/x86_64-linux-gnu/libuuid.so.1 node_modules/canvas/build/Release/libuuid.so.1
```

**Do NOT:** Set LD_LIBRARY_PATH in the workflow shell command — `/lib/x86_64-linux-gnu` in LD_LIBRARY_PATH causes `__vdso_time: invalid mode for dlopen` crash in NixOS bash.
