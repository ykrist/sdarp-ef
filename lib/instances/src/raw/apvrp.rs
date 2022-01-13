use crate::Map;

pub type Time = isize;
pub type Cost = isize;

#[derive(Debug, Clone, Default)]
pub struct Meisel {
  pub id: String,
  pub n_req: usize,
  pub n_passive: usize,
  pub n_active: usize,
  pub n_loc: usize,
  pub tmax: Time,
  pub srv_time_pickup: Vec<Time>,
  pub srv_time_delivery: Vec<Time>,
  pub start_time: Vec<Time>,
  pub end_time: Vec<Time>,
  pub compat_req_passive: Vec<Vec<usize>>,
  pub compat_passive_active: Vec<Vec<usize>>,
  pub travel_cost: Map<(usize, usize), Cost>,
  pub travel_time: Map<(usize, usize), Time>, // FIXME this should be moved to instances (dataset module), and just store coords here
}

