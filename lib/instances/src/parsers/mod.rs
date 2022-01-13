mod cordeau;
pub use cordeau::CordeauFmt;

mod meisel;
pub use meisel::MeiselFmt;

mod riedler;
pub use riedler::RiedlerFmt;


mod nom_prelude {
  pub use nom::{
    IResult, Parser,
    error::{
      self,
      ParseError,
      FromExternalError,
      context,
    },
    sequence::*,
    multi::*,
    combinator::*,
    character::complete::*,
    bytes::complete::tag,
    number::complete::double,
    Finish,
  };
  pub use std::str::FromStr;
  pub use std::num::{ParseIntError, ParseFloatError};
}

mod common;

pub trait ParseInstance<Fmt>: Sized {
  fn parse(inputs: Fmt) -> crate::Result<Self>;
}
