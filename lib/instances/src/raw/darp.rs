use crate::Map;

pub type Time = f64;
pub type Demand = isize;

pub struct Cordeau {
  pub num_requests : usize,
  pub num_vehicles : usize,
  pub max_route_time : usize,
  pub max_ride_time : usize,
  pub vehicle_capacity : usize,
  pub coords : Vec<(f64, f64)>,
  pub tw_start : Vec<Time>,
  pub tw_end : Vec<Time>,
  pub service_time : Vec<Time>,
  pub demand : Map<usize, Demand>,
}
