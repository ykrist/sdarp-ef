use super::*;
use rayon::prelude::*;
use crate::sdarp::frag::Locset;

pub trait DominationFamily {
  type Key: Hash + Eq + Copy + Send;

  fn family_key(f: &Fragment) -> Self::Key;
}

pub trait DominationCriterion: DominationFamily {
  fn select_criterion(info: &FragInfo, family_key: &Self::Key) -> PairwiseDomination;
}

pub trait DominationObj {}

pub struct Cover;
pub struct TravelTime;

impl DominationObj for Cover {}
impl DominationObj for TravelTime {}

pub struct Frag;
pub struct Ef;

pub struct SdarpDomination<F, Objective>(pub F, pub Objective);

impl<F, O> DominationFamily for SdarpDomination<F, O> {
  type Key = (Loc, Loc, Locset);

  fn family_key(f: &Fragment) -> Self::Key {
    (f.start, f.end, f.locs)
  }
}

impl DominationCriterion for SdarpDomination<Frag, Cover> {
  fn select_criterion(_info: &FragInfo, _key: &Self::Key) -> PairwiseDomination {
    domination_criteria::fragments_cover
  }
}

impl DominationCriterion for SdarpDomination<Ef, Cover> {
  fn select_criterion(info: &FragInfo, key: &Self::Key) -> PairwiseDomination {
    if key.1 == info.data.d_depot {
      domination_criteria::ef_depot_cover
    } else {
      domination_criteria::fragments_cover
    }
  }
}

impl DominationCriterion for SdarpDomination<Frag, TravelTime> {
  fn select_criterion(_info: &FragInfo, _key: &Self::Key) -> PairwiseDomination {
    domination_criteria::fragments_tt
  }
}

impl DominationCriterion for SdarpDomination<Ef, TravelTime> {
  fn select_criterion(info: &FragInfo, key: &Self::Key) -> PairwiseDomination {
    if key.1 == info.data.d_depot {
      domination_criteria::ef_depot_tt
    } else {
      domination_criteria::fragments_tt
    }
  }
}



pub type PairwiseDomination = fn(&Fragment, &Fragment) -> bool;

pub mod domination_criteria {
  use super::*;

  pub fn fragments_cover(f: &Fragment, g: &Fragment) -> bool {
    f.tef <= g.tef && f.tls >= g.tls &&
      max(f.tef, g.tls + f.ttt) <= max(g.tef, g.tls + g.ttt)
  }

  pub fn ef_depot_cover(f: &Fragment, g: &Fragment) -> bool {
    f.tls >= g.tls
  }

  pub fn fragments_tt(f: &Fragment, g: &Fragment) -> bool {
    f.tef <= g.tef && f.tls >= g.tls && f.ttt <= g.ttt
  }

  pub fn ef_depot_tt(f: &Fragment, g: &Fragment) -> bool {
    f.tls >= g.tls && f.ttt <= g.ttt
  }
}


fn dominate_family(frag_info: &FragInfo, members: Vec<Fragment>, dom_crit: PairwiseDomination) -> Vec<FragmentId> {
  let span_dg = trace_span!("dominate_group");
  let _g = span_dg.enter();

  let mut dominated_mask = vec![false; members.len()];

  for (kf, f) in members.iter().enumerate() {
    if dominated_mask[kf] { continue; }

    for (kg, g) in members.iter().enumerate().skip(kf + 1) {
      debug_assert!(kg > kf);
      let span = trace_span!("cmp", fid=f.id.raw(), gid=g.id.raw(),
                fpath=?&frag_info.paths[&f.id],gpath=?&frag_info.paths[&g.id], f=?&f, g=?&g);
      let _g = span.enter();

      if dom_crit(f, g) {
        // f dominates g
        trace!(target: "dominate", "f dominates g");
        dominated_mask[kg] = true;
      } else if dom_crit(g, f) {
        // g dominates f
        trace!(target: "dominate", "g dominates f");
        dominated_mask[kf] = true;
      }
    }
  }

  members.into_iter()
    .enumerate()
    .filter_map(|(k, f)| if dominated_mask[k] { Some(f.id) } else { None })
    .collect()
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SdarpObjKind {
  TravelTime,
  Cover
}

#[instrument(level="debug", skip(finfo))]
pub fn dominate<D: DominationCriterion>(finfo: &mut FragInfo) {
  let mut domination_groups: Map<_, _> = Map::default();

  for (_, &f) in &finfo.fragments {
    // domination_groups.entry((f.start, f.end, f.locs)).or_insert_with(Vec::new).push(f)
    domination_groups.entry(D::family_key(&f)).or_insert_with(Vec::new).push(f)
  }
  debug!(num_families=domination_groups.len());

  let to_remove: Vec<_> = domination_groups.into_par_iter()
    .map(|(key, group)|
      dominate_family(&finfo, group, D::select_criterion(&finfo, &key))
    )
    .flatten_iter()
    .collect();

  info!(removed=to_remove.len(), remaining=finfo.fragments.len()-to_remove.len(), "domination finished");
  for fid in to_remove {
    finfo.fragments.remove(&fid);
    finfo.paths.remove(&fid);
  }
}

#[inline]
fn dominate_with_obj<F>(finfo: &mut FragInfo, obj: SdarpObjKind)
  where
    SdarpDomination<F, TravelTime> : DominationCriterion,
    SdarpDomination<F, Cover> : DominationCriterion,
{
  match obj {
    SdarpObjKind::TravelTime => dominate::<SdarpDomination<F, TravelTime>>(finfo),
    SdarpObjKind::Cover => dominate::<SdarpDomination<F, Cover>>(finfo),
  }
}

#[inline]
pub fn dominate_fragments(finfo: &mut FragInfo, obj: SdarpObjKind) {
  dominate_with_obj::<Frag>(finfo, obj)
}

#[inline]
pub fn dominate_extended_fragments(finfo: &mut FragInfo, obj: SdarpObjKind) {
  dominate_with_obj::<Ef>(finfo, obj)
}

