use super::darp::{DarpInstance};
use crate::parsers::{ParseInstance, RiedlerFmt};
use crate::raw::{FromRaw, darp::Cordeau};

use super::*;

pub type SdarpInstance = DarpInstance;

pub enum SdarpRiedler {}

impl Dataset for StdLayout<SdarpRiedler> {
  type Instance = SdarpInstance;

  fn load_instance(&self, idx: usize) -> Result<Self::Instance> {
    let instance = self.index_to_name(idx)?;
    let mut path = self.dir.join(&*instance);
    path.set_extension(&self.suffix);
    let raw = Cordeau::parse(RiedlerFmt(&path)).context(format!("failed to load {:?}", path))?;
    Ok(SdarpInstance::from_raw(raw, instance))
  }
}
pub enum SdarpHard {}

impl Dataset for StdLayout<SdarpHard> {
  type Instance = SdarpInstance;

  fn load_instance(&self, idx: usize) -> Result<Self::Instance> {
    let instance = self.index_to_name(idx)?;
    let mut path = self.dir.join(&*instance);
    path.set_extension(&self.suffix);
    let raw = Cordeau::parse(RiedlerFmt(&path)).context(format!("failed to load {:?}", path))?;
    Ok(SdarpInstance::from_raw(raw, instance))
  }
}

lazy_static!{
    pub static ref RIEDLER: StdLayout<SdarpRiedler> = {
        pretty_unwrap(StdLayout::new("SDARP_riedler", "dat"))
    };

    pub static ref HARD: StdLayout<SdarpHard> = {
      pretty_unwrap(StdLayout::new("SDARP_hard", "dat"))
    };

    pub static ref DSET: DSetCollection<SdarpInstance> = {
      DSetCollection::new()
        .push_ref(&*RIEDLER)
        .push_ref(&*HARD)
        .finish()
   };
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[allow(non_snake_case)]
  fn load_30N_4K_A() -> crate::Result<()> {
    let _data = RIEDLER.load_instance(0)?;
    Ok(())
  }

  #[test]
  #[allow(non_snake_case)]
  fn load_hard_instance() -> crate::Result<()> {
    let _data = HARD.load_instance(0)?;
    Ok(())
  }
}