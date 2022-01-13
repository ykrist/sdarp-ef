use std::path::Path;
use crate::{Result, Map};
use crate::raw::darp::*;
use super::{
  ParseInstance,
  nom_prelude::*
};

#[derive(Debug, Copy, Clone)]
pub struct CordeauFmt<P>(pub P);

impl<P: AsRef<Path>> ParseInstance<CordeauFmt<P>> for Cordeau {
  fn parse(path: CordeauFmt<P>) -> Result<Cordeau> {
    let path = path.0.as_ref();
    let data = std::fs::read_to_string(path)?;
    match parsers::cordeau(&data).finish() {
      Ok((_, instance)) => Ok(instance),
      Err(e) => Err(
        anyhow::Error::msg(e.to_string())
      ),
    }
  }
}


mod parsers {
  use super::*;
  use crate::parsers::{
    common::*
  };
  use nom::number::complete as number;
  use nom::number::complete::double;

  fn time<'a, E>(input: &'a str) -> IResult<&'a str, Time, E>
    where
      E: error::ParseError<&'a str>
  {
    number::double(input)
  }

  pub fn cordeau(input: &str) -> IResult<&str, Cordeau, error::VerboseError<&str>> {
    let usize_space = |i| terminated(usize_, space1)(i);
    let time_space = |i| terminated(time, space1)(i);
    let dbl_space = |i| terminated(double, space1)(i);

    let (mut input, (num_vehicles, num_requests, max_route_time, vehicle_capacity, max_ride_time)) =
      tuple((usize_space, usize_space, usize_space, usize_space, usize_line))(input)?;

    //   7   4.832  -8.990   3   1    0 1440
    let mut parse_data_line = preceded(space0, tuple((
      usize_space, // ID
      dbl_space,  // x
      dbl_space,  // y
      time_space,  // srv time
      terminated(isize_, space1), // demand
      time_space,  // tw start
      terminated(time, newline) // td end
    )));

    let nlocs = 2*num_requests + 2;
    let mut coords = Vec::with_capacity(nlocs);
    let mut demand: Map<usize, isize> = Map::with_capacity_and_hasher(nlocs, fnv::FnvBuildHasher::default());
    let mut service_time = Vec::with_capacity(nlocs);
    let mut tw_start = Vec::with_capacity(nlocs);
    let mut tw_end = Vec::with_capacity(nlocs);

    for k in 0..nlocs {
      let (i, (loc, x, y, s, q, e, l)) = parse_data_line(input)?;
      debug_assert_eq!(loc, k);
      input = i;
      coords.push((x,y));
      if loc > 0 && loc < nlocs - 1 {
        demand.insert(loc, q);
      }
      service_time.push(s);
      tw_start.push(e);
      tw_end.push(l);
    }

    let (input, _) = eof(input)?;

    Ok((input, Cordeau {
      num_requests,
      num_vehicles,
      max_ride_time,
      max_route_time,
      vehicle_capacity,
      coords,
      demand,
      service_time,
      tw_start,
      tw_end,
    }))

  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a2_16() -> Result<()> {
    Cordeau::parse(CordeauFmt("../../data/DARP_cordeau/a2-16.txt"))?;
    Ok(())
  }
}