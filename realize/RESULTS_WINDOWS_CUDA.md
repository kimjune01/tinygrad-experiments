# Windows CUDA bench run — 2026-05-07

Cross-platform replication of the bench from a Windows host. Confirms the speedup
holds outside METAL, and documents two Windows-specific setup gotchas worth folding
into the bootstrap script.

## Host

| Field | Value |
|-------|-------|
| GPU | NVIDIA RTX 5000 Ada Generation |
| OS | Windows 11 Enterprise (10.0.26200) |
| Python | 3.14 |
| Backend | `DEV=CUDA` (driver via `nvcuda.dll`) |
| CUDA Toolkit | v12.9 |
| Model | `llama3.2:1b` (Q6_K, bartowski GGUF) |
| Bench | `python -m tinygrad.llm --model llama3.2:1b --benchmark 15` |

## Results

Steady-state tok/s (median of last ~10 of 15 tokens, after JIT warm-up):

| Branch | Steady-state | Notes |
|--------|--------------|-------|
| `master` (lazy weights) | **~10.35 tok/s** (~96.6 ms/tok) | 1080–1290 MB |
| `contiguous-prune-llm` (fix) | **~85.8 tok/s** (~11.65 ms/tok) | 2495–2758 MB |
| **Speedup** | **~8.3×** | mirrors METAL prior of ~8–12× |

Cold-path tokens (compile + first-rollout) are noisier; the gap shows up cleanly
once the rollout JIT is captured (~token 4–5 onward).

## Equivalence

`run_equiv.py` runs 20 deterministic tokens twice on the fix branch, then once on
master, all with `Tensor.manual_seed(42)` and `temperature=0.0`. All three runs
emit the same prefix:

```
[0, 475, 2808, 198, 1527, 662, 1179, 4211, 198, 1527]
```

So master ≡ fix bit-exactly on this prompt. The user's bootstrap script as written
does `master==master` (both sides re-load the current branch) which is a
determinism check, not a cross-branch check — both pass, and a manual cross-branch
run also passes.

## Windows-specific gotchas (fold into bootstrap)

1. **`DEV=NV` is hard-asserted off on Windows** (`tinygrad/runtime/ops_nv.py:3`
   `assert sys.platform != 'win32'`). The bootstrap script as written
   (`set DEV=NV`) errors before any token is generated. Use `DEV=CUDA` on
   Windows.

2. **`DEV=CUDA` cannot find the driver lib by default.** tinygrad's loader
   searches `PATH` for `cuda.dll`, but the Windows driver lib is
   `nvcuda.dll` (in `C:\Windows\System32`). Override with:

   ```bat
   set CUDA_PATH=C:\Windows\System32\nvcuda.dll
   ```

   (The loader at `tinygrad/runtime/support/c.py:95` treats `{NM}_PATH` as a
   direct file path if it points at a file, bypassing the name-based search.)
   Note this overwrites the standard NVIDIA `CUDA_PATH` directory variable for
   the shell — fine for the bench session, do it in a scoped shell.

3. **Default Python thread stack (~1 MB on Windows, ~8 MB on Linux) blows
   `pretty_print` recursion** when the kernel `__repr__` is called via
   `hashlib.sha256(str(uop))` during runtime caching (`tinygrad/uop/ops.py:147`).
   Symptom: `RecursionError: Stack overflow (used 2912 kB)` after the first
   token, with a deep `pretty_print → __repr__ → pretty_print` trace. Workaround:
   run the bench on a worker thread with a larger stack:

   ```python
   import sys, threading, runpy
   sys.setrecursionlimit(200000)
   threading.stack_size(64 * 1024 * 1024)  # 64 MiB

   def run():
       sys.argv = ['tinygrad.llm', '--model', 'llama3.2:1b', '--benchmark', '15']
       runpy.run_module('tinygrad.llm', run_name='__main__')

   t = threading.Thread(target=run); t.start(); t.join()
   ```

   This wrapper (`run_bench.py`) is what produced the numbers above. Fixing the
   underlying recursion (iterative `pretty_print`, or skipping `__repr__` inside
   the hash key) would remove the workaround need; tracked as a follow-up, not a
   blocker for the bench result itself.

## Files (in `tinygrad-test/` checkout)

| File | Purpose |
|------|---------|
| `run_bench.py` | Bench wrapper: large stack + recursion limit + invoke `tinygrad.llm` |
| `run_equiv.py` | Equivalence wrapper: same envelope, runs the determinism check |
