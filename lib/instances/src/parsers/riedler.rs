use std::path::Path;
use crate::{Result, Map};
use crate::raw::darp::*;
use super::{
  ParseInstance,
  nom_prelude::*,
};

#[derive(Debug, Copy, Clone)]
pub struct RiedlerFmt<P>(pub P);

impl<P: AsRef<Path>> ParseInstance<RiedlerFmt<P>> for Cordeau {
  fn parse(input: RiedlerFmt<P>) -> Result<Self> {
    let path = input.0.as_ref();
    let data = std::fs::read_to_string(path)?;
    match parsers::riedler(&data).finish() {
      Ok((_, instance)) => Ok(instance),
      Err(e) => Err(
        anyhow::Error::msg(e.to_string())
      ),
    }
  }
}

mod parsers {
  use super::*;
  use super::super::common::*;

  fn header_line<'a, E>(t: &str, input: &'a str) -> IResult<&'a str, usize, E>
    where
      E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>
  {
    preceded(tag(t), preceded(space0, usize_line))(input)
  }

  #[derive(Debug)]
  struct LocData {
    x: f64,
    y: f64,
    tw_start: usize,
    tw_end: usize,
  }

  fn loc_data<'a, E>(i: &'a str) -> IResult<&'a str, LocData, E>
    where
      E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError>
  {
    let (i, x) = double(i)?;
    let (i, y) = preceded(space1, double)(i)?;
    let (i, tw_start) = preceded(space1, usize_)(i)?;
    let (i, tw_end) = preceded(space1, usize_)(i)?;
    Ok((i, LocData { x, y, tw_start, tw_end }))
  }

  #[derive(Debug)]
  struct ReqData {
    pickup: LocData,
    delivery: LocData,
    srv_time: usize,
    size: usize,
  }

  fn request<'a, E>(i: &'a str) -> IResult<&'a str, ReqData, E>
    where
      E: error::ParseError<&'a str> + error::FromExternalError<&'a str, ParseIntError>
  {
    let (i, pickup) = preceded(space0, loc_data)(i)?;
    let (i, delivery) = preceded(space1, loc_data)(i)?;
    let (i, size) = preceded(space1, usize_)(i)?;
    let (i, srv_time) = preceded(space1, usize_)(i)?;
    let (i, _) = space0(i)?;
    Ok((i, ReqData { pickup, delivery, srv_time, size }))
  }

  pub fn riedler(input: &str) -> IResult<&str, Cordeau, error::VerboseError<&str>> {
    let i = input;
    let (i, n_req) = header_line("|N|:", i)?;
    let (i, n_vehicle) = header_line("|K|:", i)?;
    let (i, max_ride_time) = header_line("L:", i)?;

    let (i, _) = tag("Depot:")(i)?;
    let (i, (depot_x, depot_y, depot_e, depot_l)) = tuple((
      preceded(space0, usize_),
      preceded(space1, usize_),
      preceded(space1, usize_),
      preceded(space1, usize_line),
    ))(i)?;

    let (i, _) = tag("Vehicles\n")(i)?;
    let (i, mut vehicle_data) = separated_list_m_n(
      n_vehicle, n_vehicle, newline,
      tuple((usize_, preceded(space1, usize_))),
    )(i)?;

    let (vehicle_cap, max_route_time) = vehicle_data.pop().unwrap();

    for (a, b) in vehicle_data {
      debug_assert_eq!(a, vehicle_cap, "vehicle capacities are not identical!");
      debug_assert_eq!(b, max_route_time, "max route times are not identical!");
    }

    let (i, _) = tag("\nRequests\n")(i)?;
    let (i, requests) = separated_list_m_n(n_req, n_req, newline, request)(i)?;

    let (i, _) = many0_count(newline)(i)?;
    eof(i)?;

    let coords = {
      let mut v = Vec::with_capacity(requests.len() * 2 + 2);
      v.push((depot_x as f64, depot_y as f64));
      v.extend(requests.iter().map(|r| (r.pickup.x, r.pickup.y)));
      v.extend(requests.iter().map(|r| (r.delivery.x, r.delivery.y)));
      v.push((depot_x as f64, depot_y as f64));
      v
    };

    let tw_start = {
      let mut v = Vec::with_capacity(requests.len() * 2 + 2);
      v.push(depot_e as Time);
      v.extend(requests.iter().map(|r| r.pickup.tw_start as Time));
      v.extend(requests.iter().map(|r| r.delivery.tw_start as Time));
      v.push(depot_e as Time);
      v
    };

    let tw_end = {
      let mut v = Vec::with_capacity(requests.len() * 2 + 2);
      v.push(depot_l as Time);
      v.extend(requests.iter().map(|r| r.pickup.tw_end as Time));
      v.extend(requests.iter().map(|r| r.delivery.tw_end as Time));
      v.push(depot_l as Time);
      v
    };

    let service_time = {
      let mut v = Vec::with_capacity(requests.len() * 2 + 2);
      v.push(0 as Time);
      v.extend(requests.iter().map(|r| r.srv_time as Time));
      v.extend(requests.iter().map(|r| r.srv_time as Time));
      v.push(0 as Time);
      v
    };

    let demand : Map<_, _> = requests.iter().enumerate()
      .map(|(k, req)| (k + 1, req.size as Demand))
      .chain(requests.iter().enumerate()
        .map(|(k, req)| (k + 1 + requests.len(), -(req.size as Demand)))
      )
      .collect();

    Ok((i, Cordeau {
      num_requests: n_req,
      num_vehicles: n_vehicle,
      max_route_time,
      max_ride_time,
      vehicle_capacity: vehicle_cap,
      coords,
      tw_start,
      tw_end,
      service_time,
      demand
    }))
  }
}
