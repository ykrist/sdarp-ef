use super::*;
use lazy_static::lazy_static;
use crate::{
  parsers::{
    ParseInstance,
    MeiselFmt,
  },
  raw::{
    FromRaw,
    apvrp::{Meisel}
  },
};
use fnv::FnvHashMap as Map;

pub use crate::raw::apvrp::{Time, Cost};

pub type Loc = u16;
pub type Pv = Loc;
pub type Av = Loc;
pub type Req = Loc;

pub struct ApvrpInstance {
  pub id: String,
  pub odepot: Loc,
  pub ddepot: Loc,
  pub n_req: Loc,
  pub n_passive: Loc,
  pub n_active: Loc,
  pub n_loc: Loc,
  pub tmax: Time,
  pub srv_time: Map<Req, Time>,
  pub start_time: Map<Req, Time>,
  pub end_time: Map<Req, Time>,
  pub compat_req_passive: Map<Req, Vec<Pv>>,
  pub compat_passive_req: Map<Pv, Vec<Req>>,
  pub compat_passive_active: Map<Pv, Vec<Av>>,
  pub travel_cost: Map<(Loc, Loc), Cost>,
  pub travel_time: Map<(Loc, Loc), Time>,
}


#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct LocSetStarts {
  pub avo: Loc,
  pub pv_o: Loc,
  pub pv_d: Loc,
  pub req_p: Loc,
  pub req_d: Loc,
  pub avd: Loc,
}

impl LocSetStarts {
  pub fn new(n_passive: Loc, n_req: Loc) -> Self {
    let avo = 0;
    let pv_o = 1;
    let pv_d = pv_o + n_passive;
    let req_p = pv_d + n_passive;
    let req_d = req_p + n_req;
    let avd = req_d + n_req;
    LocSetStarts{ avo, pv_o, pv_d, req_p, req_d, avd }
  }
}


impl FromRaw<Meisel> for ApvrpInstance {
  fn from_raw(raw: Meisel, _: Cow<str>) -> ApvrpInstance {
    let ddepot = raw.n_loc as Loc;
    let odepot = 0;
    let locset_starts = LocSetStarts::new(raw.n_passive as Loc, raw.n_req as Loc);

    let srv_time = {
      let mut m = Map::default();
      for (k, &t) in raw.srv_time_pickup.iter().enumerate() {
        m.insert(k as Loc + locset_starts.req_p, t);
      }
      for (k, &t) in raw.srv_time_delivery.iter().enumerate() {
        m.insert(k as Loc + locset_starts.req_d as Loc, t);
      }
      m
    };

    let start_time = {
      let mut m = Map::default();
      for (k, &t) in raw.start_time.iter().enumerate() {
        m.insert(k as Loc + locset_starts.req_p, t);
      }
      m
    };

    let end_time = {
      let mut m = Map::default();
      for (k, &t) in raw.end_time.iter().enumerate() {
        m.insert(k as Loc + locset_starts.req_d, t);
      }
      m
    };

    let (compat_req_passive, compat_passive_req) = {
      let (mut m_r_p, mut m_p_r) = (Map::default(), Map::default());
      for (kr, pvs) in raw.compat_req_passive.iter().enumerate() {
        let r = locset_starts.req_p + kr as Loc;
        for &kp in pvs {
          let p = locset_starts.pv_o + kp as Loc;
          m_p_r.entry(p).or_insert_with(Vec::new).push(r);
          m_r_p.entry(r).or_insert_with(Vec::new).push(p);
        }
      }
      (m_r_p, m_p_r)
    };

    let compat_passive_active = {
      let mut m = Map::default();
      for (kp, avs) in raw.compat_passive_active.iter().enumerate() {
        let p = locset_starts.pv_o + kp as Loc;
        for &a in avs {
          m.entry(p).or_insert_with(Vec::new).push(a as Av)
        }
      }
      m
    };

    let map_depot_arcs = |((i, j), val)| {
      let i = i as Loc;
      let j = j as Loc;
      if j == odepot { ((i, ddepot), val) }
      else { ((i, j), val) }
    };


    ApvrpInstance {
      id: raw.id,
      odepot,
      ddepot,
      n_req: raw.n_req as Loc,
      n_passive: raw.n_passive as Loc,
      n_active: raw.n_active as Loc,
      n_loc: raw.n_loc as Loc,
      tmax: raw.tmax,
      compat_passive_active,
      compat_req_passive,
      compat_passive_req,
      start_time,
      end_time,
      srv_time,
      travel_cost: raw.travel_cost.into_iter().map(map_depot_arcs).collect(),
      travel_time: raw.travel_time.into_iter().map(map_depot_arcs).collect(),
    }
  }
}

pub enum Apvrp {}

impl Dataset for StdLayout<Apvrp> {
  type Instance = ApvrpInstance;

  fn load_instance(&self, idx: usize) -> Result<ApvrpInstance> {
    let instance = self.index_to_name(idx)?;
    let mut path = self.dir.join(&*instance);
    path.set_extension(&self.suffix);
    Meisel::parse(MeiselFmt(&path))
      .context(format!("failed to load {:?}", path))
      .map(|raw| ApvrpInstance::from_raw(raw, Cow::Borrowed("")))
  }
}

/// Apply a factor of `scale` to the travel times *and* travel costs.  Note that this will
/// apply rounding a second time, which is consistent with Michael's code.
pub fn rescale_distances(mut data: ApvrpInstance, scale: f64) -> ApvrpInstance {

  for tt in data.travel_time.values_mut() {
    *tt = (*tt as f64 * scale).round() as Time;
  }
  for tc in data.travel_cost.values_mut() {
    *tc = (*tc as f64 * scale).round() as Cost;
  }
  data
}

use crate::modify::{DSetModify, Mapped};

fn tilk_rescale(data: ApvrpInstance) -> ApvrpInstance {
  rescale_distances(data, 0.5)
}

lazy_static! {
    pub static ref TILK: StdLayout<Apvrp> = {
        pretty_unwrap(StdLayout::new("apvrp_tilk", "txt"))
    };

    pub static ref TILK_B_PART_RAW: StdLayout<Apvrp> = {
        pretty_unwrap(StdLayout::new("apvrp_tilk_b_part", "txt"))
    };


    pub static ref MEISEL: StdLayout<Apvrp> = {
        pretty_unwrap(StdLayout::new("apvrp_meisel", "txt"))
    };

    pub static ref MEISEL_A: Subset<&'static StdLayout<Apvrp>> = Subset::new(&*MEISEL, (0..30).collect());

    pub static ref TILK_AB: Mapped<Subset<&'static StdLayout<Apvrp>>, fn(ApvrpInstance) -> ApvrpInstance>
      = Subset::new(&*TILK, (0..160).collect()).map(tilk_rescale);

    pub static ref TILK_B_PART: Mapped<&'static StdLayout<Apvrp>, fn(ApvrpInstance) -> ApvrpInstance>
      = (&*TILK_B_PART_RAW).map(tilk_rescale);


    pub static ref DSET: DSetCollection<ApvrpInstance> = DSetCollection::new()
        .push_ref(&*TILK_AB)
        .push_ref(&*MEISEL_A)
        .push_ref(&*TILK_B_PART)
        .finish();
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn check_dset() -> anyhow::Result<()> {
    for idx in 0..160 {
      assert_eq!(TILK.index_to_name(idx)?, DSET.index_to_name(idx)?);
    }

    for (i,j) in (160..190).enumerate() {
      assert_eq!(MEISEL.index_to_name(i)?, DSET.index_to_name(j)?);
    }

    Ok(())
  }

  #[test]
  fn load_all() -> anyhow::Result<()> {
    for idx in 0..DSET.len() {
      DSET.load_instance(idx)?;
    }
    Ok(())
  }
}