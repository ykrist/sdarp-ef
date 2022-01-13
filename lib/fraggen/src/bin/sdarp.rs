use json;
use rayon::ThreadPoolBuilder;
use std::io::Write;
use std::str::FromStr;
use itertools::Itertools;
use anyhow::{Result};
use tracing::*;


use fraggen::*;
use fraggen::sdarp::{frag, schedule, preprocessing, ef, SdarpObjKind};
use fraggen::data::get_sdarp_instance_by_index;
use fraggen::data::darp::*;
use fraggen::sdarp::frag::{Fragment};

mod common;
use common::*;
use common::FragmentGeneration;

use structopt::StructOpt;

#[derive(Debug, Copy, Clone)]
enum GenObject {
    Fragments,
    ExtendedFragments,
}

impl FromStr for GenObject {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        return match s {
            "frags" => Ok(Self::Fragments),
            "ef" => Ok(Self::ExtendedFragments),
            _ => Err(format!("invalid string: {}", s))
        };
    }
}

fn parse_domination_kind(s: &str) -> Result<Option<SdarpObjKind>, String> {
  match s {
    "none" => Ok(None),
    "cover" => Ok(Some(SdarpObjKind::Cover)),
    "tt" => Ok(Some(SdarpObjKind::TravelTime)),
    s => Err(format!("{} is not recognised", s))
  }
}


#[derive(Debug, StructOpt)]
struct ClArgs {
    #[structopt(parse(try_from_str))]
    object: GenObject,
    #[structopt()]
    index: usize,
    #[structopt(long, short="c", default_value="1", validator=clap_range_validator(Some(1), None))]
    cpus: usize,
    #[structopt(long="no-preprocess", parse(from_flag=std::ops::Not::not))]
    preprocess: bool,
    #[structopt(long, parse(try_from_str=parse_domination_kind), possible_values=&["none", "cover", "tt"], default_value="cover")]
    domination: std::option::Option<SdarpObjKind>,
    #[structopt(flatten)]
    output: OutputOptions,
}


fn get_record(f: Fragment, path: Vec<Loc>, early_schedule: Vec<Time>, late_schedule: Vec<Time>) -> json::JsonValue {
    return json::object! {
        path: json::JsonValue::from(path),
        early_schedule: json::JsonValue::from(early_schedule),
        late_schedule: json::JsonValue::from(late_schedule),
        tls : f.tls,
        tef : f.tef,
        ttt : f.ttt,
        id : f.id.raw().get(),
    }
}

impl<'a> FragmentGeneration for frag::Network<'a> {
    fn write_json(&self, mut buf: impl Write) -> Result<()> {
        let root: json::JsonValue = self.fragments.values()
            .map(|&f| {
                let path = self.paths[&f.id].clone();
                let mut early_schedule = schedule::get_early_depots(&path, &self.data, 0);
                early_schedule.pop();
                early_schedule.remove(0);
                let mut late_schedule = schedule::get_late_depots(&path, &self.data, Time::max_value());
                late_schedule.pop();
                late_schedule.remove(0);
                return get_record(f, path, early_schedule, late_schedule);
            }
            )
            .collect_vec()
            .into();

        root.write_pretty(&mut buf, 2)?;
        return Ok(())
    }

    fn write_json_summary(&self, mut buf : impl Write) -> Result<()> {
        let root: json::JsonValue = self.size_info.clone().into();
        root.write_pretty(&mut buf, 2)?;
        return Ok(())
    }

}

impl<'a> FragmentGeneration for ef::Network<'a> {
    fn write_json(&self, mut buf: impl Write) -> Result<()> {
        let root: json::JsonValue = self.ef.values()
            .map(|&f| {
                let path = &self.paths[&f.id];

                let (early_schedule, late_schedule) = if *path.last().unwrap() == self.data.d_depot {
                    let mut depot_path = Vec::with_capacity(path.len()+1);
                    depot_path.push(self.data.o_depot);
                    depot_path.extend_from_slice(path);
                    let mut es = schedule::get_early(&depot_path, &self.data, 0);
                    es.remove(0);
                    let mut ls = schedule::get_late(&depot_path, &self.data, Time::max_value());
                    ls.remove(0);
                    (es,ls)
                } else if path[0] == self.data.o_depot {
                    debug_assert_eq!(path.len(), 2);
                    let mut depot_path = Vec::with_capacity(4);
                    depot_path.extend_from_slice(path);
                    depot_path.push(self.data.dmap(path[1]));
                    depot_path.push(self.data.d_depot);
                    let mut es = schedule::get_early(&depot_path, &self.data, 0);
                    es.pop();
                    es.pop();
                    let mut ls = schedule::get_late(&depot_path, &self.data, Time::max_value());
                    ls.pop();
                    ls.pop();
                    (es,ls)
                } else {
                    let mut depot_path = Vec::with_capacity(path.len() + 3);
                    depot_path.push(self.data.o_depot);
                    depot_path.extend_from_slice(path);
                    depot_path.push(self.data.dmap(*path.last().unwrap()));
                    depot_path.push(self.data.d_depot);
                    let mut es = schedule::get_early(&depot_path, &self.data, 0);
                    es.pop();
                    es.pop();
                    let mut ls = schedule::get_late(&depot_path, &self.data, Time::max_value());
                    ls.pop();
                    ls.pop();
                    (es,ls)
                };

                return get_record(f, path.clone(), early_schedule, late_schedule);
            }
            )
            .collect_vec()
            .into();

        root.write_pretty(&mut buf, 2)?;
        return Ok(())
    }

    fn write_json_summary(&self, mut buf : impl Write) -> Result<()> {
        let root: json::JsonValue = self.size_info.clone().into();

        root.write_pretty(&mut buf, 2)?;
        return Ok(())
    }
}


fn main() -> anyhow::Result<()> {
    let args : ClArgs = StructOpt::from_args();
    let _g = init_logging(args.output.log.clone());
    debug!(?args);
    ThreadPoolBuilder::new().num_threads(args.cpus).build_global().expect("Failed to construct thread pool");
    let mut data = get_sdarp_instance_by_index(args.index).unwrap();
    if args.preprocess {
        preprocessing::preprocess(&mut data);
    }
    let data = data;

  match args.object {
    GenObject::Fragments => {
      let network = frag::build_network(&data, args.domination);
      output_network(&args.output, network)?;
    },
    GenObject::ExtendedFragments => {
      let network = ef::build_network(&data, args.domination);
      output_network(&args.output, network)?
    },
  }
  Ok(())
}