use anyhow::Result;
use instances::dataset::{sdarp, Dataset, IdxNameMap};

pub mod darp;

pub fn get_sdarp_instance_by_name(name : &str) -> Result<darp::DarpInstance> {
  get_sdarp_instance_by_index(sdarp::DSET.name_to_index(name)?)
}


pub fn get_sdarp_instance_by_index(idx : usize) -> Result<darp::DarpInstance> {
    sdarp::DSET.load_instance(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn fail_load_sdarp_instance() {
        get_sdarp_instance_by_name("non-existent").unwrap();
    }

    #[test]
    #[should_panic]
    fn fail_load_sdarp_instance_idx() {
        get_sdarp_instance_by_index(999).unwrap();
    }

}