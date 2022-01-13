use std::fmt;
use std::iter::FromIterator;
use std::cmp::max;
use std::collections::HashMap;
use itertools::Itertools;
use tracing::*;
use rayon::prelude::*;

use crate::*;
use crate::data::darp::*;
use crate::utils::Biterator;
use super::{schedule, FragmentId, SdarpObjKind, dominate_fragments};

const LOCSET_WORDS: usize = 2;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Locset([u128; LOCSET_WORDS]);

impl Locset {
  pub fn new() -> Self {
    return Self([0u128; LOCSET_WORDS]);
  }

  pub fn iter<'a>(&'a self) -> impl Iterator<Item=Loc> + 'a {
    self.0.iter()
      .enumerate()
      .map(|(k, &bits)| Biterator::new(bits).map(move |i| i as Loc + ((k as Loc) << 7)))
      .flatten()
  }

  pub fn to_vec(&self) -> Vec<Loc> { 
    self.iter().collect()
   }

  #[inline]
  pub fn insert(&mut self, i: Loc) {
    let (word_index, bit_index) = Self::word_bit_index(&i);
    self.0[word_index] |= 1 << bit_index;
  }

  #[inline]
  fn word_bit_index(i: &Loc) -> (usize, u8) {
    let word_index = (i >> 7) as usize; // divide by 128 = 2**7, rounding down;
    let bit_index = i & 0x7f; // modulo 128
    return (word_index, bit_index);
  }

  #[inline]
  pub fn contains(&self, i: &Loc) -> bool {
    let (word_index, bit_index) = Self::word_bit_index(i);
    return (self.0[word_index] & (1 << bit_index)) != 0;
  }

  pub fn is_disjoint(&self, other: &Self) -> bool {
    self.0.iter().zip(other.0.iter()).all(|(x, y)| x & y == 0)
  }

  pub fn union(&self, other: &Self) -> Self {
    let mut new = self.clone();
    new.union_inplace(other);
    return new;
  }

  pub fn union_inplace(&mut self, other: &Self) {
    self.0.iter_mut().zip(other.0.iter()).for_each(|(x, y)| *x |= y);
  }
}


impl fmt::Debug for Locset {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_set()
      .entries(self.iter())
      .finish()
  }
}

impl<T: Into<u128>> FromIterator<T> for Locset {
  fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
    let mut words = [0u128; LOCSET_WORDS];
    for i in iter {
      let i: u128 = i.into();
      let word_index = (i >> 7) as usize; // divide by 128 = 2**7, rounding down;
      let bit_index = i & 0x7f; // modulo 128
      words[word_index] |= 1 << bit_index;
    }
    return Self(words);
  }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Fragment {
  pub start: Loc,
  pub end: Loc,
  pub id: FragmentId,
  pub locs: Locset,
  pub tef: Time,
  pub tls: Time,
  pub ttt: Time,
}

impl Fragment {
  pub fn new(start: Loc, end: Loc, locs: Locset, tef: Time, tls: Time, ttt: Time) -> Self {
    let id = FragmentId::new();
    return Fragment {
      start,
      end,
      id,
      locs,
      tef,
      tls,
      ttt,
    };
  }
}


fn extend(data: &DarpInstance,
          pathlist: &mut Vec<(Vec<Loc>, Time)>,
          path: Vec<Loc>,
          time: Time,
          load: Demand,
          pending_deliveries: Vec<(Loc, Time)>,
          total_tt: Time) -> bool {
  let m = path.len();
  let i = path[m - 1];

  if time > data.tw_end[&i]
    || total_tt + *data.travel_time.get(&(i, data.d_depot)).unwrap_or(&0) > data.max_ride_time[&data.o_depot]
    || load > data.capacity
    || pending_deliveries.iter().any(|&(d, t)| t > data.max_ride_time[&data.pmap(d)] + *data.travel_time.get(&(i, d)).unwrap_or(&0)) {
    return false;
  }

  if pending_deliveries.is_empty() {
    match schedule::try_get_early_depots(&path, data, 0) {
      None => return false,
      Some(schedule) => {
        trace!(?path, "fragment");
        pathlist.push((path, schedule[schedule.len() - 2]));
        return true;
      }
    }
  }

  let mut can_finish = false;
  for &(j, _) in &pending_deliveries {
    if let Some(&tt) = data.travel_time.get(&(i, j)) {
      let new_pending_deliveries = pending_deliveries.iter()
        .cloned()
        .filter_map(|(d, t)| if d == j { None } else { Some((d, t + tt)) })
        .collect();
      let mut new_path = path.clone();
      new_path.push(j);

      can_finish |= extend(
        data,
        pathlist,
        new_path,
        max(time + tt, data.tw_start[&j]),
        load + data.demand[&j],
        new_pending_deliveries,
        total_tt + tt,
      );
    }
  }

  if !can_finish { return false; }

  for &j in &data.P {
    if path.contains(&j) { continue; }

    if let Some(&tt) = data.travel_time.get(&(i, j)) {
      let mut new_pending_deliveries: Vec<_> = pending_deliveries.iter()
        .cloned()
        .map(|(d, t)| (d, t + tt))
        .collect();
      new_pending_deliveries.push((data.dmap(j), 0));

      let mut new_path = path.clone();
      new_path.push(j);
      extend(
        data,
        pathlist,
        new_path,
        max(time + tt, data.tw_start[&j]),
        load + data.demand[&j],
        new_pending_deliveries,
        total_tt + tt,
      );
    }
  }

  return true;
}


pub struct FragInfo<'a> {
  pub data: &'a DarpInstance,
  pub fragments: Map<FragmentId, Fragment>,
  pub paths: Map<FragmentId, Vec<Loc>>,
}

#[instrument(level = "info", skip(data))]
pub(super) fn generate_fragments<'a>(data: &'a DarpInstance) -> FragInfo<'a> {
  let fragment_paths: Vec<_> = data.P.par_iter()
    .map(|&p| {
      let _s = info_span!("frag_gen_job", ?p).entered();
      let mut pathlist = Vec::new();
      let path = vec![p];
      let time = data.tw_start[&p];
      let load = data.demand[&p];
      let pending_deliveries = vec![(data.dmap(p), 0)];
      extend(data, &mut pathlist, path, time, load, pending_deliveries, data.travel_time[&(data.o_depot, p)]);
      return pathlist.into_iter();
    })
    .flatten_iter()
    .collect();

  let fragment_info: Vec<_> = fragment_paths.into_par_iter()
    .map(|(path, tef)| {
      let late = schedule::get_late_depots(&path, &data, Time::max_value());
      let tls = late[1];
      let ttt: Time = path.iter().tuple_windows().map(|(&i, &j)| data.travel_time[&(i, j)]).sum();
      let locs: Locset = path.iter().cloned().collect();
      let f = Fragment::new(path[0], *path.last().unwrap(), locs, tef, tls, ttt);
      return (f, path);
    })
    .collect();

  let mut fragments = Map::default();
  let mut paths = Map::default();
  for (f, path) in fragment_info {
    trace!(?path,?f, "create fragment");
    paths.insert(f.id, path);
    fragments.insert(f.id, f);
  }

  info!(count = fragments.len(), "{:?} fragments generated", fragments.len());
  return FragInfo { data, fragments, paths };
}


pub struct Network<'a> {
  pub data: &'a DarpInstance,
  pub fragments: Map<FragmentId, Fragment>,
  pub paths: Map<FragmentId, Vec<Loc>>,
  pub size_info: HashMap<String, isize>,
}

#[instrument(level="info",name="build_frag_network",  skip(data))]
pub fn build_network<'a>(data: &'a DarpInstance, domination: Option<SdarpObjKind>) -> Network<'a> {

  let mut finfo = generate_fragments(data);
  let mut size_info = HashMap::default();
  size_info.insert("fragments".to_string(), finfo.fragments.len() as isize);
  if let Some(obj) = domination {
    dominate_fragments(&mut finfo, obj);
    size_info.insert("undominated_fragments".to_string(), finfo.fragments.len() as isize);
  }

  return Network {
    data,
    fragments: finfo.fragments,
    paths: finfo.paths,
    size_info,
  };
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::data::get_sdarp_instance_by_index;
  use crate::init_test_logging;
  use anyhow::Result;

  const LOGFILE: Option<&str> = Some("testlog.ndjson");

  #[test]
  fn fragment_size() {
    init_test_logging(None::<&str>);
    info!("size of Fragment = {} bytes", std::mem::size_of::<Fragment>());
    assert!(std::mem::size_of::<Fragment>() <= 64)
  }

  #[test]
  fn gen_frags() -> Result<()> {
    // init_test_logging(None::<&str>);
    let _g = init_test_logging(LOGFILE);
    let data = get_sdarp_instance_by_index(0)?;
    info!(Q=?&data.demand);
    let frags = generate_fragments(&data);
    assert_eq!(frags.paths.len(), frags.fragments.len());
    return Ok(());
  }

  #[test]
  fn locset_insert() {
    let mut locs = Locset::new();
    locs.insert(0);
    assert_eq!(locs.to_vec(), vec![0]);
  }

  #[test]
  fn locset_from_iter() {
    fn check(vec: Vec<Loc>) {
      let locs: Locset = vec.iter().cloned().collect();
      assert_eq!(vec, locs.to_vec())
    }
    check(vec![0]);
    check(vec![0, 9, 89, 99]);
    check(vec![0, 1, 2, 3, 5, 128, 255]);
  }


  // fn _test_triangle_inequality(index : usize) {
  // TODO need to split paths properly
  //
  //     let span = trace_span!("test_triangle_ineq", index);
  //     let _g = span.enter();
  //
  //     let data = get_sdarp_instance_by_index(index).unwrap();
  //     let network = build_network(&data, false);
  //
  //     let paths : HashSet<_> = network.paths.values().cloned().collect();
  //     // trace!(?paths);
  //     for path in &paths {
  //         if path.len() <= 3 {
  //             continue;
  //         }
  //         let s = trace_span!("drop_requests", ?path);
  //         let _g = s.enter();
  //
  //         for &p in path {
  //             if data.is_pickup(p) {
  //                 let new_path : Vec<_> = path.iter()
  //                     .filter_map(|&i| if i == p || i == data.dmap(p) { None } else {Some(i)})
  //                     .collect();
  //
  //                 trace!(new_path=?new_path);
  //                 if !paths.contains(&new_path) {
  //                     let legal = schedule::check_depots(&new_path, &data, 0);
  //                     error!(new_path=?new_path,legal, "missing fragment");
  //                     panic!("bug")
  //                 }
  //             }
  //         }
  //     }
  // }
  //
  // #[test]
  // fn triangle_inequality() {
  //     let _g = init_test_logging(LOGFILE);
  //     _test_triangle_inequality(0);
  // }
}


