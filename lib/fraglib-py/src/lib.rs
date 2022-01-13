use pyo3::prelude::*;

pub mod apvrp;
pub mod sdarp;

pub(crate) fn run_with_threadpool<T: Send>(cpus: usize, func: impl Send + FnOnce() -> PyResult<T>) -> PyResult<T> {
  let cpus = if cpus > 0 { cpus } else { num_cpus::get_physical() };
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(cpus)
    .build()
    .map_err(|e| pyo3::exceptions::PyException::new_err(format!("Failed to construct thread pool: {}", e.to_string())))?;
  return pool.install(func);
}

// IMPORTANT: function must be what we import in Python
#[pymodule]
fn fraglibpy(py: Python, m: &PyModule) -> PyResult<()> {
  #[cfg(feature = "logging")] {
    fraggen::init_logging(None::<&str>);
  }
  m.add_submodule(apvrp::build_module(py)?)?;
  m.add_submodule(sdarp::build_module(py)?)?;
  Ok(())
}
