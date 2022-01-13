use pyo3::prelude::*;
use pyo3::{wrap_pyfunction};
use pyo3::exceptions::PyValueError;
use fnv::FnvHashMap as Map;
use std::ops::{Deref, DerefMut};
use std::collections::HashMap;
use crate::run_with_threadpool;

use instances::dataset::{
  Dataset,
  sdarp::*,
  darp::{
    Loc,
    Demand,
    Time,
    Cost,
  },
  IdxNameMap,
};
use fraggen::sdarp::SdarpObjKind;


#[pyclass(module = "sdarp")]
pub struct SdarpDataWrapper {
  inner: SdarpInstance,
}

impl Deref for SdarpDataWrapper {
  type Target = SdarpInstance;
  fn deref(&self) -> &SdarpInstance {
    &self.inner
  }
}

impl DerefMut for SdarpDataWrapper {
  fn deref_mut(&mut self) -> &mut SdarpInstance {
    &mut self.inner
  }
}

#[allow(non_snake_case)]
#[pymethods(module = "sdarp")]
impl SdarpDataWrapper {
  #[getter("id")]
  fn id(&self) -> String { self.id.clone() }

  #[getter("n")]
  fn n(&self) -> Loc { self.n.clone() }

  #[getter("num_req")]
  fn num_req(&self) -> usize { self.P.len() }

  #[getter("num_vehicles")]
  fn num_vehicles(&self) -> usize { self.K.len() }

  #[getter("capacity")]
  fn capacity(&self) -> Demand { self.capacity }


  #[getter("tw_start")]
  fn tw_start(&self) -> Map<Loc, Time> {
    self.tw_start.clone()
  }

  #[getter("tw_end")]
  fn tw_end(&self) -> Map<Loc, Time> {
    self.tw_end.clone()
  }

  #[getter("max_ride_time")]
  fn max_ride_time(&self) -> Map<Loc, Time> {
    self.max_ride_time.clone()
  }

  #[getter("travel_time")]
  fn travel_time(&self) -> Map<(Loc, Loc), Time> {
    self.travel_time.clone()
  }

  #[getter("travel_cost")]
  fn travel_cost(&self) -> Map<(Loc, Loc), Cost> {
    self.travel_cost.clone()
  }

  #[getter("demand")]
  fn demand(&self) -> Map<Loc, Demand> {
    self.demand.clone()
  }

  #[getter("o_depot")]
  fn o_depot(&self) -> Loc {
    self.o_depot
  }

  #[getter("d_depot")]
  fn d_depot(&self) -> Loc {
    self.d_depot
  }
}


impl SdarpDataWrapper {
  pub fn new(instance: SdarpInstance) -> Self {
    SdarpDataWrapper { inner: instance }
  }
}


#[pyfunction]
pub fn preprocess_data(data: &mut SdarpDataWrapper) -> PyResult<()> {
  fraggen::sdarp::preprocessing::preprocess(&mut *data);
  Ok(())
}

pub type Erf = (Vec<Loc>, Time, Time, Time);

#[pyfunction("*", cpus=0)]
pub fn extended_restricted_fragments(data: &SdarpDataWrapper, domination: Option<&str>, cpus: usize) -> PyResult<(Vec<Erf>, HashMap<String, isize>)> {
  let domination = match domination {
    None => None,
    Some("cover") => Some(SdarpObjKind::Cover),
    Some("tt") => Some(SdarpObjKind::TravelTime),
    Some(_) => return Err(PyErr::new::<PyValueError, _>("value must be `cover` or `tt`"))
  };
  run_with_threadpool(cpus, || {
    let network = fraggen::sdarp::ef::build_network(&data, domination);
    let info = network.size_info.clone();
    let frags : Vec<Erf> = network.ef.iter()
      .map(|(fid, f)| (network.paths[fid].clone(), f.tef, f.tls, f.ttt))
      .collect();
    return Ok((frags, info))
  })
}



mod instance {
  use super::*;

  macro_rules! impl_idx_name_fn {
    () => {
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
    };
}

  impl_idx_name_fn! {}

  #[pyfunction]
  pub fn load(idx: usize) -> PyResult<SdarpDataWrapper> {
    DSET.load_instance(idx)
      .map_err(|e| pyo3::exceptions::PyException::new_err(e.to_string()))
      .map(SdarpDataWrapper::new)
  }

  pub fn build_module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "instance")?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(name_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_name, m)?)?;
    m.add_function(wrap_pyfunction!(len, m)?)?;
    Ok(m)
  }
}


pub(crate) fn build_module(py: Python) -> PyResult<&PyModule> {
  let m = PyModule::new(py, "sdarp")?;
  m.add_submodule(instance::build_module(py)?)?;
  m.add_class::<SdarpDataWrapper>()?;
  m.add_function(wrap_pyfunction!(extended_restricted_fragments, m)?)?;
  m.add_function(wrap_pyfunction!(preprocess_data, m)?)?;
  Ok(m)
}
