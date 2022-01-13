use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use lazy_static::lazy_static;
use crate::Error;
use fnv::FnvHashSet;
use std::borrow::Cow;


pub trait IdxNameMap {
  fn index_to_name(&self, idx: usize) -> Result<Cow<str>>;

  fn name_to_index(&self, name: &str) -> Result<usize>;

  fn len(&self) -> usize;

  fn check_idx(&self, idx: usize) -> Result<()> {
    if self.len() <= idx {
      Err(Error::IndexOutOfRange.into())
    } else {
      Ok(())
    }
  }
}


impl<'a, D: IdxNameMap> IdxNameMap for &'a D {
  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    D::index_to_name(self, idx)
  }

  fn name_to_index(&self, name: &str) -> Result<usize> {
    D::name_to_index(self, name)
  }

  fn len(&self) -> usize {
    D::len(self)
  }
}

pub trait Dataset: IdxNameMap + Sync {
  type Instance;
  fn load_instance(&self, idx: usize) -> Result<Self::Instance>;
}


impl<'a, D: Dataset> Dataset for &'a D {
  type Instance = D::Instance;

  fn load_instance(&self, idx: usize) -> Result<Self::Instance> {
    D::load_instance(self, idx)
  }
}


/// A Standard Layout Dataset: a directory containing instance files and `INDEX.txt` index file.
/// The index file contains a new-line separated list of instance names, which acts as a map from index -> name.
/// Each instance file is named `NAME.SUFFIX`.
pub struct StdLayout<D> {
  _marker: PhantomData<D>,
  name_order: Vec<String>,
  name_to_idx_map: HashMap<String, usize>,
  dir: PathBuf,
  suffix: String,
}


impl<D> StdLayout<D> {
  fn new(dir: impl AsRef<Path>, suffix: &str) -> Result<StdLayout<D>> {
    let root = std::env::var("DATA_ROOT").expect("environment variable DATA_ROOT must be defined");
    let dir = Path::new(&root).join(dir);
    let ctx = format!("try read directory {:?}", &dir);
    let dir = dir.canonicalize().context(ctx)?;

    let contents = std::fs::read_to_string(dir.join("INDEX.txt"))?;
    let name_order: Vec<String> = contents.split_whitespace().map(|s| s.trim().to_string()).collect();
    let name_to_idx_map: HashMap<_, _> = name_order.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    Ok(StdLayout {
      _marker: PhantomData {},
      name_order,
      name_to_idx_map,
      dir,
      suffix: suffix.to_string(),
    })
  }
}

impl<D> IdxNameMap for StdLayout<D> {
  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    self.check_idx(idx)?;
    Ok(Cow::Borrowed(&self.name_order[idx]))
  }

  fn name_to_index(&self, name: &str) -> Result<usize> {
    self.name_to_idx_map.get(name).ok_or(Error::UnkownInstanceName.into()).map(|i| *i)
  }

  fn len(&self) -> usize { self.name_order.len() }
}


/// A Standard Layout Dataset: a directory containing instance files and `INDEX.txt` index file.
/// The index file contains a new-line separated list of instance names, which acts as a map from index -> name.
/// Each instance file is named `NAME.SUFFIX`.
pub struct DynLayout<D> {
  _marker: PhantomData<D>,
  name_order: Vec<PathBuf>,
  name_to_idx_map: HashMap<String, usize>,
}

impl<D> DynLayout<D> {
    #[allow(dead_code)]
  fn new(dir: impl AsRef<Path>, patt: &str) -> Result<Self> {
    let root = std::env::var("DATA_ROOT").expect("environment variable DATA_ROOT must be defined");
    let dir = Path::new(&root).join(dir);

    let mut p = dir.to_string_lossy().into_owned();
    p.push('/');
    p.push_str(patt);

    let names : std::result::Result<Vec<PathBuf>, _> = glob::glob(&p)?.collect();
    let name_order = names?;
    let name_to_idx_map: Result<HashMap<_, _>> = name_order.iter()
      .enumerate()
      .map(|(k, p)| {
        let n = p.file_stem().ok_or_else(|| anyhow::anyhow!("missing file stem: {:?}", p))?;
        Ok((n.to_string_lossy().into_owned(), k))
      })
      .collect();
    let name_to_idx_map = name_to_idx_map?;
    Ok(DynLayout {
      _marker: Default::default(),
      name_order,
      name_to_idx_map
    })
  }
}

impl<D> IdxNameMap for DynLayout<D> {
  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    self.check_idx(idx)?;
    let name = self.name_order[idx].file_stem()
      .ok_or_else(|| anyhow::anyhow!("missing file stem for idx {}", idx))?;
    Ok(name.to_string_lossy())
  }

  fn name_to_index(&self, name: &str) -> Result<usize> {
    let idx = *self.name_to_idx_map.get(name).ok_or(Error::UnkownInstanceName)?;
    Ok(idx)
  }

  fn len(&self) -> usize { self.name_order.len() }
}


#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum DSetIdx {
  Owned(usize),
  Ref(usize),
}


pub struct DSetCollectionBuilder<I: 'static> {
  static_refs: Vec<&'static dyn Dataset<Instance=I>>,
  owned: Vec<Box<dyn Dataset<Instance=I>>>,
  order: Vec<DSetIdx>,
}

impl<I> DSetCollectionBuilder<I> {
  pub fn push_owned<D: Dataset<Instance=I> + 'static>(mut self, dset: D) -> Self {
    self.order.push(DSetIdx::Owned(self.owned.len()));
    self.owned.push(Box::new(dset));
    self
  }

  pub fn push_ref<D: Dataset<Instance=I>>(mut self, dset: &'static D) -> Self {
    self.order.push(DSetIdx::Ref(self.static_refs.len()));
    self.static_refs.push(dset);
    self
  }

  pub fn finish(self) -> DSetCollection<I> {
    let DSetCollectionBuilder { static_refs, order, owned } = self;
    let length = static_refs.iter()
        .map(|d| d.len())
        .chain(owned.iter().map(|d| d.len()))
        .sum::<usize>();

    let mut name_to_idx = HashMap::with_capacity(length);
    let mut offset = 0;
    let mut dset_lengths = Vec::with_capacity(order.len());

    for didx in &order {
      let dset = match didx {
        DSetIdx::Owned(idx) => owned[*idx].as_ref(),
        DSetIdx::Ref(idx) => static_refs[*idx],
      };

      for i in 0..dset.len() {
        let old = name_to_idx.insert(
          dset.index_to_name(i).unwrap().to_string(),
          i + offset,
        );
        assert_eq!(old, None, "duplicate instance name in collection");
      }
      dset_lengths.push(dset.len());
      offset += dset.len();
    }

    DSetCollection { order, owned, static_refs, dset_lengths, length, name_to_idx }
  }
}


pub struct DSetCollection<I: 'static> {
  static_refs: Vec<&'static dyn Dataset<Instance=I>>,
  owned: Vec<Box<dyn Dataset<Instance=I>>>,
  order: Vec<DSetIdx>,
  dset_lengths: Vec<usize>,
  name_to_idx: HashMap<String, usize>,
  length: usize,
}

impl<I> DSetCollection<I> {
  pub fn new() -> DSetCollectionBuilder<I> {
    DSetCollectionBuilder { static_refs: Vec::new(), owned: Vec::new(), order: Vec::new() }
  }

  fn get_dset(&self, didx: DSetIdx) -> &dyn Dataset<Instance=I> {
    match didx {
      DSetIdx::Owned(idx) => self.owned[idx].as_ref(),
      DSetIdx::Ref(idx) => self.static_refs[idx],
    }
  }

  fn map_index(&self, idx: usize) -> Result<(DSetIdx, usize)> {
    let mut offset = 0;
    for (&dlen, didx) in self.dset_lengths.iter().zip(&self.order) {
      offset += dlen;
      if idx < offset {
        return Ok((*didx, idx - (offset - dlen)));
      }
    }
    return Err(Error::IndexOutOfRange.into());
  }
}

impl<I> IdxNameMap for DSetCollection<I> {
  fn name_to_index(&self, name: &str) -> Result<usize> {
    self.name_to_idx.get(name)
      .ok_or(Error::UnkownInstanceName.into())
      .map(|i| *i)
  }

  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    let (didx, idx) = self.map_index(idx)?;
    self.get_dset(didx).index_to_name(idx)
  }

  fn len(&self) -> usize { self.length }
}

impl<I> Dataset for DSetCollection<I> {
  type Instance = I;

  fn load_instance(&self, idx: usize) -> Result<I> {
    let (didx, idx) = self.map_index(idx)?;
    self.get_dset(didx).load_instance(idx)
  }
}

pub struct Subset<D> {
  dataset: D,
  indices: Vec<usize>,
  index_set: FnvHashSet<usize>,
}

impl<D: IdxNameMap> Subset<D> {
  pub fn new(dataset: D, indices: Vec<usize>) -> Self {
    for &i in &indices {
      if i >= dataset.len() {
        panic!("index {} out of range (0..{})", i, dataset.len())
      }
    }
    let index_set: FnvHashSet<_> = indices.iter().cloned().collect();
    if index_set.len() != indices.len() {
      panic!("indices must be unique")
    }
    Subset { dataset, indices, index_set }
  }

  fn map_index(&self, idx: usize) -> Result<usize> {
    self.indices.get(idx).copied().ok_or_else(|| Error::IndexOutOfRange.into())
  }
}

impl<D: IdxNameMap> IdxNameMap for Subset<D> {
  fn name_to_index(&self, name: &str) -> Result<usize> {
    let idx = self.dataset.name_to_index(name)?;
    if self.index_set.contains(&idx) {
      Ok(idx)
    } else {
      Err(Error::UnkownInstanceName.into())
    }
  }

  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    self.dataset.index_to_name(self.map_index(idx)?)
  }

  fn len(&self) -> usize { self.indices.len() }
}

impl<I, D: Dataset<Instance=I>> Dataset for Subset<D> {
  type Instance = I;
  fn load_instance(&self, idx: usize) -> Result<I> {
    self.dataset.load_instance(self.map_index(idx)?)
  }
}


pub mod apvrp;
pub mod darp;
pub mod sdarp;


fn pretty_unwrap<T>(r: Result<T>) -> T {
  match r {
    Err(e) => panic!("{:?}", e),
    Ok(t) => t
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn datasubset() {
    let subset = Subset::new(&*apvrp::DSET, vec![0, 4, 1]);
    assert_eq!(subset.len(), 3);
    assert_eq!(subset.map_index(0).unwrap(), 0);
    assert_eq!(subset.map_index(1).unwrap(), 4);
    assert_eq!(subset.map_index(2).unwrap(), 1);
    subset.load_instance(2).unwrap();
  }

  #[test]
  fn load_one() {
    pretty_unwrap(apvrp::DSET.load_instance(1));
    pretty_unwrap(darp::DSET.load_instance(1));
  }
}

