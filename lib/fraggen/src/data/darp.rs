pub use instances::dataset::darp::{
  Time,
  Cost,
  Vehicle,
  Demand,
  Loc,
};

pub type DarpInstance = instances::dataset::darp::DarpInstance;

pub trait DarpInstanceExt {
  fn is_pickup(&self, i: Loc) -> bool;
  fn is_delivery(&self, i: Loc) -> bool;
  fn dmap(&self, i: Loc) -> Loc;
  fn pmap(&self, i: Loc) -> Loc;
}

impl DarpInstanceExt for DarpInstance {
    #[inline]
    fn is_pickup(&self, i: Loc) -> bool {
        return 0 < i && i <= self.n;
    }

    #[inline]
    fn is_delivery(&self, i: Loc) -> bool {
        return self.n < i && i <= self.n * 2;
    }

    #[inline]
    fn dmap(&self, i: Loc) -> Loc {
        if i == self.o_depot {
            return self.d_depot;
        } else {
            debug_assert!(self.is_pickup(i));
            return i + self.n;
        }
    }

    #[inline]
    fn pmap(&self, i: Loc) -> Loc {
        if i == self.d_depot {
            return self.o_depot;
        } else {
            debug_assert!(self.is_delivery(i));
            return i - self.n;
        }
    }
}