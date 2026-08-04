---
name: Imported dependency preservation
description: Preserve an imported project's declared dependency ranges during Replit setup.
---

When setting up an imported project, installing dependencies may rewrite package manifests to newer direct dependency ranges even when the project already has a lockfile. Restore the imported declarations unless an upgrade is explicitly part of the request.

**Why:** Setup should make the project runnable without silently changing the dependency contract or introducing unrelated upgrade risk.

**How to apply:** Capture the original manifest before installing. After installation, compare the manifest and lockfile; keep only the minimum changes needed for reproducible startup.