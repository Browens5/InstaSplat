use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Select clone/split indices from densify scores (AbsGS / gsplat-style).
///
/// Returns ``(clone_idx, split_idx)`` as Python lists of int.
#[pyfunction]
#[pyo3(signature = (grads, scale_max, grad_threshold, clone_scale, phase, room))]
fn select_clone_split<'py>(
    py: Python<'py>,
    grads: PyReadonlyArray1<'py, f32>,
    scale_max: PyReadonlyArray1<'py, f32>,
    grad_threshold: f32,
    clone_scale: f32,
    phase: &str,
    room: usize,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let g = grads.as_slice()?;
    let s = scale_max.as_slice()?;
    if g.len() != s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grads and scale_max length mismatch",
        ));
    }
    let n = g.len();
    let mut clone_cand: Vec<(f32, i64)> = Vec::new();
    let mut split_cand: Vec<(f32, i64)> = Vec::new();
    for i in 0..n {
        if g[i] < grad_threshold {
            continue;
        }
        if s[i] <= clone_scale {
            clone_cand.push((g[i], i as i64));
        } else {
            split_cand.push((g[i], i as i64));
        }
    }
    clone_cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    split_cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let (clone_budget, split_budget) = if room == 0 {
        (0, 0)
    } else {
        match phase {
            "split" => {
                let sb = split_cand.len().min(room);
                let quarter = (clone_cand.len() / 4).max(1);
                let cb = clone_cand.len().min(room.saturating_sub(sb)).min(quarter);
                (cb, sb)
            }
            "clone" => {
                let cb = clone_cand.len().min(room);
                let quarter = (split_cand.len() / 4).max(1);
                let sb = split_cand
                    .len()
                    .min(room.saturating_sub(cb))
                    .min(quarter);
                (cb, sb)
            }
            _ => {
                let cb = clone_cand.len().min((room + 1) / 2);
                let sb = split_cand.len().min(room.saturating_sub(cb));
                (cb, sb)
            }
        }
    };

    let clone_idx: Vec<i64> = clone_cand.into_iter().take(clone_budget).map(|(_, i)| i).collect();
    let split_idx: Vec<i64> = split_cand.into_iter().take(split_budget).map(|(_, i)| i).collect();
    Ok((PyList::new_bound(py, clone_idx), PyList::new_bound(py, split_idx)))
}

#[pymodule]
fn instasplat_densify(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(select_clone_split, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
