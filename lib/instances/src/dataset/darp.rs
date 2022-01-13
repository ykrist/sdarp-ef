use super::*;
use crate::parsers::{ParseInstance, CordeauFmt};
use crate::raw::darp::Cordeau;
use crate::Map;
use crate::raw::{
  metrics::{Euclidean, Metric},
  FromRaw
};

use itertools::Itertools;

pub type Time = u32;
pub type Loc = u8;
pub type Vehicle = u8;
pub type Cost = i32;
pub type Demand = i8;

pub const TIME_PREC : u32 = 5;
pub const TIME_SCALE : f64 = 100_000.0; // 10**TIME_PREC
pub const COST_PREC : u32 = TIME_PREC;
pub const COST_SCALE : f64 = TIME_SCALE;


#[allow(non_snake_case)]
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct DarpInstance {
  pub id: String,
  pub n: Loc,
  pub P: Vec<Loc>,
  pub D: Vec<Loc>,
  pub K: Vec<Vehicle>,
  pub N: Vec<Loc>,
  pub capacity: Demand,
  pub travel_time: Map<(Loc, Loc), Time>,
  pub travel_cost: Map<(Loc, Loc), Cost>,
  pub demand: Map<Loc, Demand>,
  pub tw_start: Map<Loc, Time>,
  pub tw_end: Map<Loc, Time>,
  pub max_ride_time: Map<Loc, Time>, // includes max route time (max_ride_time[0])
  pub o_depot: Loc,
  pub d_depot: Loc,
}

impl FromRaw<Cordeau> for DarpInstance {
  fn from_raw(raw: Cordeau, id: Cow<str>) -> DarpInstance {
    let n = raw.num_requests as Loc;
    let pickups: Vec<Loc> = (1..=n).collect();
    let locations: Vec<Loc> = (0..=2*n+1).collect();
    let o_depot: Loc = 0;
    let d_depot = n * 2 + 1;

    let distance = |i : Loc, j : Loc|{
      let i = i as usize;
      let j = j as usize;
      Euclidean::compute(raw.coords[i], raw.coords[j])
    };

    let round_time = |t : f64| (t*TIME_SCALE).round() as Time;
    let round_cost = |t : f64| (t*COST_SCALE).round() as Cost;


    let mut max_ride_time : Map<_,_> = pickups.iter()
      .map(|&p| (p, round_time(raw.max_ride_time as f64 + raw.service_time[p as usize])))
      .collect();
    max_ride_time.insert(o_depot, round_time(raw.max_route_time as f64));

    let travel_time: Map<_,_> = locations.iter()
      .cartesian_product(locations.iter())
      .map(|(&i,&j)| ((i,j), round_time(distance(i, j) + raw.service_time[i as usize]))) // FIXME check this won't overflow
      .collect();

    let travel_cost: Map<_,_> = locations.iter()
      .cartesian_product(locations.iter())
      .map(|(&i,&j)| ((i,j), round_cost(distance(i, j))))// FIXME check for overflow
      .collect();

    let tw_start : Map<_,_> = locations.iter().map(|&i| (i, round_time(raw.tw_start[i as usize]))).collect();
    let tw_end : Map<_,_> = locations.iter().map(|&i| (i, round_time(raw.tw_end[i as usize]))).collect();

    DarpInstance{
      id: id.into_owned(),
      n,
      P: pickups,
      D: (n + 1..=2*n as Loc).collect(),
      N: locations,
      K: (0..raw.num_vehicles as Vehicle).collect(),
      capacity: raw.vehicle_capacity as Demand,
      travel_time,
      travel_cost,
      tw_start,
      tw_end,
      max_ride_time,
      demand : raw.demand.into_iter().map(|(i,q)| (i as Loc, q as Demand)).collect(),
      o_depot,
      d_depot,
    }
  }
}

pub fn widen_time_windows(mut data: DarpInstance, additional: f64) -> DarpInstance {
  let delay = (TIME_SCALE * 15.0 * additional).round() as Time;
  for l in data.tw_end.values_mut() {
    *l += delay;
  }
  data
}

pub fn extend_time_windows(data: DarpInstance) -> DarpInstance {
  widen_time_windows(data, 1.0)
}

pub enum DarpCordeau {}

impl Dataset for StdLayout<DarpCordeau> {
  type Instance = DarpInstance;

  fn load_instance(&self, idx: usize) -> Result<Self::Instance> {
    let instance = self.index_to_name(idx)?;
    let mut path = self.dir.join(&*instance);
    path.set_extension(&self.suffix);
    let raw = Cordeau::parse(CordeauFmt(&path)).context(format!("failed to load {:?}", path))?;
    Ok(DarpInstance::from_raw(raw, instance))
  }
}

use crate::modify::{Mapped, DSetModify};

lazy_static!{
    pub static ref CORDEAU_ORIG: StdLayout<DarpCordeau> = {
        pretty_unwrap(StdLayout::new("DARP_cordeau", "txt"))
    };

    pub static ref DSET: Mapped<&'static StdLayout<DarpCordeau>, fn(DarpInstance) -> DarpInstance> = {
    (&*CORDEAU_ORIG).map(extend_time_windows)
    };
}