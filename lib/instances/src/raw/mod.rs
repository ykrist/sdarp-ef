pub mod apvrp;
pub mod darp;
use std::borrow::Cow;

pub trait FromRaw<T> where Self: Sized {
  fn from_raw(raw: T, id: Cow<str>) -> Self;
}


pub(crate) mod metrics {
  use num_traits::{AsPrimitive, Num};
  use fnv::FnvHashMap;

  pub trait Metric {
    const SYM: bool = false;

    fn compute<T: Num + AsPrimitive<f64>>(p1: (T, T), p2: (T, T)) -> f64;
  }


  pub struct Euclidean();

  impl Metric for Euclidean {
    const SYM: bool = true;

    fn compute<T: Num + AsPrimitive<f64>>(p1: (T, T), p2: (T, T)) -> f64 {
      let a = p1.0.as_() - p2.0.as_();
      let b = p1.1.as_() - p2.1.as_();
      (a*a + b*b).sqrt()
    }
  }

  /// Compute the distance-matrix for the given coordinates
  #[inline]
  #[allow(dead_code)]
  pub fn dist_matrix<M, T>(_metric: M, coords: &[(T, T)]) -> FnvHashMap<(usize, usize), f64>
    where
      M: Metric,
      T: Num + AsPrimitive<f64>
  {
    dist_matrix_pp(_metric, coords, |x| x)
  }

  /// Like [`dist_matrix`], but allows a post-processing function to be supplied.
  pub fn dist_matrix_pp<M, T, S>(_metric: M, coords: &[(T, T)], func: impl Fn(f64) -> S) -> FnvHashMap<(usize, usize), S>
    where
      M: Metric,
      T: Num + AsPrimitive<f64>,
      S: Copy
  {
    let mut matrix = FnvHashMap::default();
    let n = coords.len();
    if M::SYM {
      for i in 0..n {
        let p1 = coords[i];
        for j in (i+1)..n {
          let p2 = coords[j];
          let d = func(M::compute(p1, p2));
          matrix.insert((i,j), d);
          matrix.insert((j,i), d);
        }
        let d = func(M::compute(p1, p1));
        matrix.insert((i,i), d);
      }
    } else {
      for i in 0..n {
        let p1 = coords[i];
        for j in 0..n {
          let p2 = coords[j];
          let d = func(M::compute(p1, p2));
          matrix.insert((i,j), d);
        }
      }
    }

    matrix
  }

}