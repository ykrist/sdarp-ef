use std::cmp::{min, max};
use itertools::Itertools;
use tracing::*;

use crate::data::darp::*;
use crate::Set;
use crate::sdarp::schedule;

#[instrument(level="debug", skip(data))]
fn tighten_time_windows(data: &mut DarpInstance) {
    for &p in &data.P {
        let d = data.dmap(p);
        let ride_time = data.max_ride_time[&p];

        let t = max(
            data.tw_start[&d].saturating_sub(ride_time),
            data.tw_start[&data.o_depot] + data.travel_time[&(data.o_depot, p)],
        );
        let tw_start_p = data.tw_start.get_mut(&p).unwrap();
        #[cfg(debug_assertions)]
            if *tw_start_p < t {trace!(p,old=*tw_start_p,new=t,"tighten TW LB")}
        *tw_start_p = max(*tw_start_p, t);
        let tw_start_p = *tw_start_p;


        let t = min(
            data.tw_end[&p] + ride_time,
            data.tw_end[&data.d_depot] - data.travel_time[&(d, data.d_depot)],
        );
        let tw_end_d = data.tw_end.get_mut(&d).unwrap();
        #[cfg(debug_assertions)]
            if *tw_end_d > t {trace!(d,old=*tw_end_d,new=t,"tighten TW UB")}
        *tw_end_d = min(*tw_end_d, t);
        let tw_end_d = *tw_end_d;

        let tt = data.travel_time[&(p, d)];

        let t = tw_end_d - tt;
        let tw_end_p = data.tw_end.get_mut(&p).unwrap();
        #[cfg(debug_assertions)]
            if *tw_end_p > t {trace!(tw_end_d,p,old=*tw_end_p,new=t,"tighten TW UB")}
        *tw_end_p = min(*tw_end_p, t);

        let t = tw_start_p + tt;
        let tw_start_d = data.tw_start.get_mut(&d).unwrap();
        #[cfg(debug_assertions)]
            if *tw_start_d < t {trace!(d,old=*tw_start_d,new=t,"tighten TW LB")}
        *tw_start_d = max(*tw_start_d, t)
    }
}


fn remove_arcs(data: &mut DarpInstance) {
    let parent_span = span!(Level::DEBUG, "remove_arcs");
    let _g = parent_span.enter();

    let mut illegal_arcs : Set<_> = data.P.iter().tuple_combinations()
        .flat_map(|(&pi, &pj)| {
            let s = span!(parent: &parent_span, Level::TRACE, "test_pair", i=pi,j=pj);
            let _g = s.enter();

            let (di, dj) = (data.dmap(pi), data.dmap(pj));

            let is_path_illegal = if data.demand[&pi] + data.demand[&pj] > data.capacity {
                let mut illegal = Vec::with_capacity(6);
                illegal.extend_from_slice(&[true; 4]);
                illegal.extend(vec![
                    vec![pi, di, pj, dj],
                    vec![pj, dj, pi, di],
                    ].into_iter()
                    .map(|path| !schedule::check_depots(&path, data, 0))
                );
                illegal
            } else {
                vec![
                    vec![pi, pj, dj, di],
                    vec![pi, pj, di, dj],
                    vec![pj, pi, di, dj],
                    vec![pj, pi, dj, di],
                    vec![pi, di, pj, dj],
                    vec![pj, dj, pi, di],
                ].into_iter()
                    .map(|path| !schedule::check_depots(&path, data, 0))
                    .collect()
            };

            trace!(?is_path_illegal);

            let mut ij_illegal_arcs : Vec<(Loc,Loc)> = Vec::with_capacity(8);

            if is_path_illegal[0] && is_path_illegal[1] { ij_illegal_arcs.push((pi, pj)); }
            if is_path_illegal[2] && is_path_illegal[3] { ij_illegal_arcs.push((pj, pi)); }
            if is_path_illegal[1] && is_path_illegal[2] { ij_illegal_arcs.push((di, dj)); }
            if is_path_illegal[0] && is_path_illegal[3] { ij_illegal_arcs.push((dj, di)); }

            if is_path_illegal[1] { ij_illegal_arcs.push((pj, di)); }
            if is_path_illegal[3] { ij_illegal_arcs.push((pi, dj)); }
            if is_path_illegal[4] { ij_illegal_arcs.push((di, pj)); }
            if is_path_illegal[5] { ij_illegal_arcs.push((dj, pi)); }

            trace!(?ij_illegal_arcs);

            return ij_illegal_arcs.into_iter()
        })
        .collect();


    for &i in &data.N {
        illegal_arcs.insert((i, i));
        illegal_arcs.insert((data.d_depot, i));
        illegal_arcs.insert((i, data.o_depot));
    }

    for &p in &data.P {
        let d = data.dmap(p);
        illegal_arcs.insert((d, p));
        illegal_arcs.insert((p, data.d_depot));
        illegal_arcs.insert((data.o_depot, d));
    }

    let num_arcs_initial = data.travel_time.len();
    for arc in illegal_arcs.iter() {
        trace!(?arc, "removed");
        data.travel_time.remove(arc);
        data.travel_cost.remove(arc);
    }
    debug!("Removed {} arcs, {} remaining", num_arcs_initial - data.travel_time.len(), data.travel_time.len());

}


pub fn preprocess(data: &mut DarpInstance) {
    let s = span!(Level::DEBUG, "preprocess", data_id=?data.id);
    let _g = s.enter();
    tighten_time_windows(data);
    remove_arcs(data);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::get_sdarp_instance_by_index;
    use crate::init_test_logging;
    use proptest::prelude::*;

    const LOGFILE : Option<&str> = Some("testlog.ndjson");

    #[test]
    fn test_preprocess() {
        let _g = init_test_logging(LOGFILE);
        let mut data=  get_sdarp_instance_by_index(0).unwrap();
        preprocess(&mut data);
    }


    #[test]
    fn test_remove_arcs() {
        let _g = init_test_logging(LOGFILE);
        let mut data=  get_sdarp_instance_by_index(0).unwrap();
        remove_arcs(&mut data);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        /// Multiple tests in here to save the expensive loading of data
        fn group_test(idx in 1..29usize) {
            init_test_logging(None::<&str>);
            let data_template = get_sdarp_instance_by_index(idx).unwrap();
            info!(data_id=?&data_template.id,idx);

            let n_arcs_no_tighten = {   // Property: Removing arcs twice shouldn't make a difference
                let mut data = data_template.clone();
                remove_arcs(&mut data);
                let n_arcs_before : usize = data.travel_time.len();
                remove_arcs(&mut data);
                prop_assert_eq!(n_arcs_before, data.travel_time.len());
                prop_assert_eq!(data.travel_cost.len(), data.travel_time.len());
                n_arcs_before
            };

            { // pipeline test
                let mut data = data_template.clone();
                tighten_time_windows(&mut data);
                remove_arcs(&mut data);
                // Property - tightening time windows shouldn't affect how many arcs were removed.
                prop_assert_eq!(data.travel_time.len(), n_arcs_no_tighten);
            }
        }
    }
}