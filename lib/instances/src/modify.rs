use anyhow::Result;
use super::dataset::*;
use std::borrow::Cow;

pub trait DSetModify<F>: Sized {
  fn map(self, func: F) -> Mapped<Self, F>;
}

impl<D, F, I, O> DSetModify<F> for D
  where
    D: Dataset<Instance=I>,
    F: Fn(I) -> O + Sync,
{
  fn map(self, func: F) -> Mapped<D, F> { Mapped{ input: self, map: func } }
}

pub struct Mapped<D, F> {
  input: D,
  map: F,
}

impl<D, F, I> IdxNameMap for Mapped<D, F>
  where
    D: Dataset<Instance=I>,
{
  #[inline]
  fn name_to_index(&self, name: &str) -> Result<usize> {
    self.input.name_to_index(name)
  }

  #[inline]
  fn index_to_name(&self, idx: usize) -> Result<Cow<str>> {
    self.input.index_to_name(idx)
  }

  #[inline]
  fn len(&self) -> usize { self.input.len() }
}

impl<D, F, I, O> Dataset for Mapped<D, F>
  where
    D: Dataset<Instance=I>,
    F: Fn(I) -> O + Sync,
{
  type Instance = O;

  fn load_instance(&self, idx: usize) -> Result<O> {
    Ok((self.map)(self.input.load_instance(idx)?))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn trivial_map() -> Result<()> {
    use crate::dataset::darp::DSET as DARP;
    let q = (&*DARP).map(|_| 0);
    assert_eq!(q.load_instance(0)?, 0);
    Ok(())
  }
}
