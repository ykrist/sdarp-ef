use std::collections::HashMap;
use std::cmp::max;
use tracing::*;
use rayon::prelude::*;

use crate::*;
use crate::data::darp::*;
use super::frag::{Fragment, FragInfo, Locset};
use super::{schedule, frag, FragmentId, SdarpObjKind, dominate_extended_fragments};

#[instrument(level="trace", skip(data))]
fn extend_fragment(data: &DarpInstance, f: Fragment, path: Vec<Loc>) -> Vec<(Fragment, Vec<Loc>)> {
    let mut extended_fragments = Vec::with_capacity(data.P.len());
    for &p in &data.P {
        let span = trace_span!("extend_test", p);
        let _g = span.enter();
        if !f.locs.contains(&p) {
            if let Some(&tt) = data.travel_time.get(&(*path.last().unwrap(), p)) {
                if f.tef + tt  <= data.tw_end[&p] {
                    let mut new_path = Vec::with_capacity(path.len() + 4);
                    new_path.push(data.o_depot);
                    new_path.extend_from_slice(&path);
                    new_path.extend_from_slice(&[p, p + data.n, data.d_depot]);
                    let l = new_path.len();
                    let schedule = schedule::get_early(&new_path, data, 0);
                    let tef = schedule[l - 3]; // need to trim both the delivery and the d_depot
                    let schedule = schedule::get_late(&new_path, data, Time::MAX);
                    let tls = schedule[1]; // first pickup
                    new_path = new_path[1..l-2].to_vec();
                    let ttt = f.ttt + data.travel_time[&(*path.last().unwrap(), p)];
                    let ef = Fragment::new(f.start, p + data.n, f.locs.clone(), tef, tls, ttt); // locs excludes the last loc
                    extended_fragments.push((ef, new_path))
                }
            }
        }
    }
    let mut depot_path = path.clone();
    depot_path.push(data.d_depot);
    let mut locs = f.locs.clone();
    locs.insert(data.d_depot);
    let ef = Fragment::new(f.start, data.d_depot, locs, Time::MAX, f.tls, f.ttt + data.travel_time[&(f.end, data.d_depot)]);
    extended_fragments.push((ef, depot_path));

    return extended_fragments;
}


pub struct Network<'a> {
    pub data: &'a DarpInstance,
    pub ef: Map<FragmentId, Fragment>,
    pub paths: Map<FragmentId, Vec<Loc>>,
    pub size_info: HashMap<String, isize>,
}

#[instrument(level="info", name="build_ef_network", skip(data))]
pub fn build_network<'a>(data: &'a DarpInstance, domination: Option<SdarpObjKind>) -> Network<'a>
{
    let frag_network = frag::build_network(data, domination);
    let (mut size_info, frags, paths) = (frag_network.size_info, frag_network.fragments, frag_network.paths);

    let extended_frags: Vec<_> = frags.into_par_iter()
        .map(|(fid, f)| extend_fragment(data, f, paths[&fid].clone()).into_iter())
        .flatten_iter()
        .collect();

    let n_start_frags = data.n as isize;
    size_info.insert("ef".to_string(), extended_frags.len() as isize + n_start_frags);

    let mut finfo = {
        let mut fragments = Map::default();
        let mut paths = Map::default();
        for (f, path) in extended_frags {
            paths.insert(f.id, path);
            fragments.insert(f.id, f);
        }
        FragInfo { data, fragments, paths }
    };

    info!(count=finfo.fragments.len(), "finished generation");
    if let Some(obj) = domination {
        dominate_extended_fragments(&mut finfo, obj);
        size_info.insert("undominated_ef".to_string(), finfo.fragments.len() as isize + n_start_frags);
    }

    for &p in &data.P {
        let tef = max(data.tw_start[&data.o_depot] + data.travel_time[&(data.o_depot, p)], data.tw_start[&p]);
        let path = vec![data.o_depot, p];
        let locs : Locset = path.iter().cloned().collect();
        let f = Fragment::new(data.o_depot, p, locs, tef, Time::MAX, data.travel_time[&(data.o_depot, p)]);
        finfo.paths.insert(f.id, path);
        finfo.fragments.insert(f.id, f);
    }

    return Network {
        data,
        ef: finfo.fragments,
        paths: finfo.paths,
        size_info
    }
}

#[cfg(test)]
mod tests {
    use crate::init_test_logging;
    use super::*;
    use crate::data::get_sdarp_instance_by_index;
    use std::collections::HashSet;

    #[allow(dead_code)]
    const LOG_FILE : Option<&str> = Some("testlog.ndjson");
    #[allow(dead_code)]
    const NO_LOG_FILE : Option<&str> = None;


    fn _test_triangle_inequality(index : usize) {
        let span = trace_span!("test_triangle_ineq", index);
        let _g = span.enter();

        let data = get_sdarp_instance_by_index(index).unwrap();
        let network = build_network(&data, None);

        let paths : HashSet<_> = network.paths.values().cloned().collect();

        for path in &paths {
            if path.len() <= 3 {
                continue;
            }
            let s = trace_span!("drop_requests", ?path);
            let _g = s.enter();

            for &p in &path[..path.len()-1] {
                if data.is_pickup(p) {
                    let new_path : Vec<_> = path.iter()
                        .filter_map(|&i| if i == p || i == data.dmap(p) { None } else {Some(i)})
                        .collect();

                    trace!(new_path=?new_path);
                    if !paths.contains(&new_path) {
                        let mut new_path_d = new_path.clone();
                        if *new_path_d.last().unwrap() == data.d_depot {
                            new_path_d.pop();
                        } else {
                            new_path_d.push(data.dmap(*new_path.last().unwrap()));
                        }

                        let legal = schedule::check_depots(&new_path_d, &data, 0);
                        error!(new_path=?new_path,legal, "missing fragment");
                        panic!("bug")
                    }
                }
            }
        }
    }

    #[test]
    fn regression_0() {
        let _g = init_test_logging(LOG_FILE);
        let data = get_sdarp_instance_by_index(0).unwrap();
        let network = build_network(&data, None);
        let paths : HashSet<_> = network.paths.values().cloned().collect();
        assert!(paths.contains(&vec![18, 16, 48, 4, 46, 23, 34, 53, 6]));
    }
}