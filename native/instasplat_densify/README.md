# instasplat_densify (optional Rust)

Fast AbsGS-style clone/split **index selection** for InstaSplat densify.

Tensor math (prune/clone/split, Adam surgery) stays in PyTorch on MPS/CPU —
moving full parameter tensors through Rust would be slower. This crate only
accelerates the score → index ranking step.

## Build

```bash
pip install maturin
maturin develop --release -m native/instasplat_densify/Cargo.toml
```

Python automatically falls back to a pure-PyTorch implementation of
`select_clone_split` if the extension is missing.
