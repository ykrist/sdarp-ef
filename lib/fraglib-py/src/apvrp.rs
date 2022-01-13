use pyo3::prelude::*;
use pyo3::{wrap_pyfunction};
use fnv::FnvHashMap as Map;
use std::ops::Deref;

use instances::dataset::{Dataset, apvrp::*, IdxNameMap};

#[pyclass(module = "apvrp")]
pub struct ApvrpDataWrapper {
  inner: ApvrpInstance,
}

impl Deref for ApvrpDataWrapper {
  type Target = ApvrpInstance;
  fn deref(&self) -> &ApvrpInstance {
    &self.inner
  }
}

#[allow(non_snake_case)]
#[pymethods(module = "apvrp")]
impl ApvrpDataWrapper {
  #[getter("id")]
  fn id(&self) -> String { self.id.clone() }

  #[getter("odepot")]
  fn odepot(&self) -> Loc { self.odepot }

  #[getter("ddepot")]
  fn ddepot(&self) -> Loc { self.ddepot }

  #[getter("n_req")]
  fn n_req(&self) -> Loc { self.n_req }

  #[getter("n_passive")]
  fn n_passive(&self) -> Pv { self.n_passive }

  #[getter("n_active")]
  fn n_active(&self) -> Av { self.n_active }

  #[getter("n_loc")]
  fn n_loc(&self) -> Loc { self.n_loc }

  #[getter("tmax")]
  fn tmax(&self) -> Time { self.tmax }

  #[getter("srv_time")]
  fn srv_time(&self) -> Map<Req, Time> { self.srv_time.clone() }

  #[getter("end_time")]
  fn end_time(&self) -> Map<Req, Time> { self.end_time.clone() }

  #[getter("start_time")]
  fn start_time(&self) -> Map<Req, Time> { self.start_time.clone() }

  #[getter("compat_req_passive")]
  fn compat_req_passive(&self) -> Map<Req, Vec<Pv>> { self.compat_req_passive.clone() }

  #[getter("compat_passive_req")]
  fn compat_passive_req(&self) -> Map<Pv, Vec<Req>> { self.compat_passive_req.clone() }

  #[getter("compat_passive_active")]
  fn compat_passive_active(&self) -> Map<Pv, Vec<Av>> { self.compat_passive_active.clone() }

  #[getter("travel_cost")]
  fn travel_cost(&self) -> Map<(Loc, Loc), Cost> { self.travel_cost.clone() }

  #[getter("travel_time")]
  fn travel_time(&self) -> Map<(Loc, Loc), Time> { self.travel_time.clone() }
}


impl ApvrpDataWrapper {
  pub fn new(instance: ApvrpInstance) -> Self {
    ApvrpDataWrapper { inner: instance }
  }
}



mod instance {
  use super::*;

  #[pyfunction]
  pub fn load(idx: usize) -> PyResult<ApvrpDataWrapper> {
    DSET.load_instance(idx)
      .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))
      .map(ApvrpDataWrapper::new)
  }


  #[pyfunction]
  pub fn index_to_name(idx: usize) -> PyResult<String> {
    DSET.index_to_name(idx)
      .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))
      .map(String::from)
  }

  #[pyfunction]
  pub fn name_to_index(name: &str) -> PyResult<usize> {
    DSET.name_to_index(name)
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  #[pyfunction]
  pub fn len() -> usize {
    DSET.len()
  }

  pub(crate) fn build_module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "instance")?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(name_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_name, m)?)?;
    m.add_function(wrap_pyfunction!(len, m)?)?;
    Ok(m)
  }
}



pub(crate) fn build_module(py: Python) -> PyResult<&PyModule> {
  let m = PyModule::new(py, "apvrp")?;
  m.add_class::<ApvrpDataWrapper>()?;
  m.add_submodule(instance::build_module(py)?)?;
  Ok(m)
}
