pub use anyhow::Result;

use std::fmt;
use fnv::FnvHashMap as Map;

#[derive(Debug, Copy, Clone)]
enum Error {
    UnkownInstanceName,
    IndexOutOfRange,
}


impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {}


pub mod dataset;
pub mod modify;
pub mod raw;

mod parsers;
