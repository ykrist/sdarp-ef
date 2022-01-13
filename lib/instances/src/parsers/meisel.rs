use super::*;
use common::*;
use nom_prelude::*;
use std::path::Path;

use crate::raw::apvrp::{
  Meisel,
  Time,
};

#[derive(Debug, Copy, Clone)]
pub struct MeiselFmt<P>(pub P);

impl<P: AsRef<Path>> ParseInstance<MeiselFmt<P>> for crate::raw::apvrp::Meisel {
  fn parse(input: MeiselFmt<P>) -> crate::Result<Self> {
    let path = input.0.as_ref();
    let data = std::fs::read_to_string(path)?;
    match parsers::meisel(&data).finish() {
      Ok((_, instance)) => Ok(instance),
      Err(e) => Err(
        anyhow::Error::msg(e.to_string())
      ),
    }
  }
}

mod parsers {
  use super::*;
  use crate::raw::metrics;

  fn time<'a, E>(input: &'a str) -> IResult<&'a str, Time, E>
    where
      E: error::ParseError<&'a str> + error::FromExternalError<&'a str, ParseIntError>
  {
    map_res(digit1, Time::from_str)(input)
  }


  pub fn meisel(input: &str) -> IResult<&str, Meisel, error::VerboseError<&str>> {
    let (i, id) = context("instance name",
                          map(terminated(alphanumeric1, newline), String::from))(input)?;

    let (i, (n_req, n_passive, n_active, n_loc)) = context("instance size",
                                                           tuple((usize_line, usize_line, usize_line, usize_line)))(i)?;

    let (i, tmax) = context("time horizon",
                            terminated(time, newline))(i)?;
    let space_comma_sep = |i| tuple((space0, char(','), space0))(i);
    let req_time_vec =
      |i| terminated(
        separated_list_m_n(n_req, n_req, space_comma_sep, time),
        newline,
      )(i);
    let (i, srv_time_pickup) = context("service time (pickup)", req_time_vec)(i)?;
    let (i, srv_time_delivery) = context("service time (delivery)", req_time_vec)(i)?;
    let (i, start_time) = context("start times", req_time_vec)(i)?;
    let (i, end_time) = context("end times", req_time_vec)(i)?;


    let (i, _) = tuple((not_line_ending, line_ending))(i)?; // skip header
    let passive_vehicle = context("passive vehicle", map(preceded(char('p'), usize_), |s| s - 1));
    let passive_vehicle_vec = terminated(
      separated_list_m_n(1, n_passive, space_comma_sep, passive_vehicle),
      newline,
    );
    let (i, compat_req_passive) = context("req-passive compatibility",
                                          many_m_n(n_req, n_req, passive_vehicle_vec))(i)?;


    let (i, _) = tuple((not_line_ending, line_ending))(i)?; // skip header
    let active_vehicle = map(preceded(char('v'), usize_), |s| s - 1);
    let active_vehicle_vec = terminated(
      separated_list_m_n(1, n_active, space_comma_sep, active_vehicle),
      newline,
    );
    let (i, compat_passive_active) = context("passive-active compatibility",
                                             many_m_n(n_passive, n_passive, active_vehicle_vec))(i)?;

    let (i, _) = tuple((not_line_ending, line_ending))(i)?; // skip
    let coord_vec = |i| terminated(
      separated_list_m_n(n_loc, n_loc, space_comma_sep, usize_),
      newline,
    )(i);
    let (i, x) = context("x-coordinates", coord_vec)(i)?;
    let (i, y) = context("y-coordinates", coord_vec)(i)?;
    let coords: Vec<_> = x.into_iter().zip(y).collect();

    let travel_time = metrics::dist_matrix_pp(metrics::Euclidean(), &coords, |d| d.round() as Time);
    let travel_cost = travel_time.clone();

    eof(i)?;

    Ok((i, Meisel {
      id,
      n_req,
      n_passive,
      n_active,
      n_loc,
      tmax,
      srv_time_pickup,
      srv_time_delivery,
      start_time,
      end_time,
      compat_req_passive,
      compat_passive_active,
      travel_cost,
      travel_time,
    }))
  }
}
