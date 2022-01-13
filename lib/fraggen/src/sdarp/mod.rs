use std::cmp::{max, min};
use tracing::*;

use crate::*;
use crate::data::darp::*;
use crate::sdarp::frag::{Fragment, FragInfo};

pub mod ef;
pub mod frag;
pub mod preprocessing;
pub mod dom;

pub use dom::{
  dominate_extended_fragments,
  dominate_fragments,
  SdarpObjKind,
};

define_nonzero_u32_id_type!(FragmentId);

pub mod schedule {
    use super::*;
    use itertools::Itertools;
    use crate::Map;

    #[inline]
    pub fn get_late_depots(path: &[Loc], data: &DarpInstance, finish_time : Time) -> Vec<Time> {
        let path = append_depots(path, data);
        return get_late(&path, data, finish_time)
    }

    /// Returns the latest schedule for the path which starts service at the last location at
    /// `finish_time` or earlier.  Does not append depots to `path` or check for legality.
    #[instrument(skip(data))]
    pub fn get_late(path: &[Loc], data: &DarpInstance, finish_time: Time) -> Vec<Time> {
        let tw_end = path.iter().map(|i| data.tw_end[i]).collect_vec();
        // travel_time[m] is the travel time from path[m] to path[m+1]
        let travel_time = {
            let mut tt = Vec::with_capacity(path.len()-1);
            for (&i,&j) in path.iter().tuple_windows() {
                match data.travel_time.get(&(i,j)) {
                    None => {
                        error!(i,j,pass=0,"no physical arc");
                        panic!("bug - late schedule should exist")
                    },
                    Some(&t) => tt.push(t),
                }
            }
            tt
        };

        // path is assumed to have depots included
        debug_assert_eq!(path[0], data.o_depot);

        let mut departure =  vec![Time::max_value(); path.len()];
        let mut service_start= vec![Time::max_value(); path.len()];

        // Pass 1 (backward pass)
        let m = path.len()-1; // index of location i in arc (i,j)

        // Maps p -> index of d(p) and d -> index of p(d)
        let mut pd_map = Map::with_capacity_and_hasher(path.len(), Default::default());
        pd_map.insert(data.pmap(path[m]), m);

        departure[m] = min(data.tw_end[&path[m]], finish_time);
        service_start[m] = departure[m];
        for m in (0..path.len()-1).rev() {
            departure[m] = service_start[m+1] - travel_time[m];
            service_start[m] = min(departure[m], tw_end[m]);
            debug_assert!(service_start[m] >= data.tw_start[&path[m]]);

            // Build lookup while we're looping
            let i = path[m];

            if data.is_delivery(i) {
                pd_map.insert(i-data.n, m);
            } else {
                pd_map.insert(data.dmap(i), m);
            }

        }
        let pd_map = pd_map;

        // Pass 2 (forward pass)
        let mut cum_waiting_time = departure[0] - service_start[0];
        // m is index of j in arc (i,j)
        for m in 1..path.len() {
            service_start[m] = departure[m-1] + travel_time[m-1];
            let j = path[m];
            if data.is_delivery(j) {
                debug_assert!(pd_map.contains_key(&j));
                if let Some(&m_p) = pd_map.get(&j) { // In RF this may return None
                    let delta = service_start[m] as i64 - service_start[m_p] as i64 - data.max_ride_time[&data.pmap(j)] as i64;
                    if delta > 0 {
                        debug_assert!(delta <= cum_waiting_time as i64);

                        service_start[m] -= delta as Time;
                        cum_waiting_time -= delta as Time;
                        // trace!(?service_start, ?travel_time);
                        for q in (0..m).rev() {
                            // trace!(q);
                            service_start[q] = min(service_start[q], service_start[q+1] - travel_time[q]);
                        }
                    }
                }
                debug_assert!(service_start[m] >= data.tw_start[&j]);
            }
            cum_waiting_time += departure[m] - service_start[m]
        }

        // Pass 3 (2nd backward pass)
        for m in (0..path.len()-1).rev() {
            departure[m] = service_start[m+1] - travel_time[m];
            service_start[m] = min(service_start[m], departure[m]);


            let i = path[m];
            debug_assert!(service_start[m] >= data.tw_start[&i]);
            #[cfg(debug_assertions)]
                {
                    if data.is_pickup(i) {
                        if let Some(&m_d) = pd_map.get(&i) {
                            if service_start[m_d] - service_start[m] > data.max_ride_time[&i] {
                                error!(schedule=?service_start, "ride time violation");
                                panic!("bug - schedule should be legal")
                            }
                        }
                    }
                }
        }

        return service_start
    }

    /// Returns the earliest schedule for the path which starts service at the first location at
    /// `start_time` or later. Does not append depots to `path` or check for legality.
    #[inline]
    pub fn get_early(path: &[Loc], data: &DarpInstance, start_time: Time) -> Vec<Time> {
        return try_get_early(path, data, start_time).unwrap()
    }

    #[inline]
    pub fn get_early_depots(path: &[Loc], data: &DarpInstance, start_time: Time) -> Vec<Time> {
        return try_get_early(&append_depots(path, data), data, start_time).unwrap()
    }

    /// Check for the existence of a legal schedule, assuming the vehicle starts at `start_time`.
    /// Does not append depots.
    #[instrument(skip(data))]
    pub fn try_get_early(path: &[Loc], data: &DarpInstance, start_time: Time) -> Option<Vec<Time>> {
        let tw_start = path.iter().map(|i| data.tw_start[i]).collect_vec();
        let tw_end = path.iter().map(|i| data.tw_end[i]).collect_vec();
        let travel_time = {
            let mut tt = Vec::with_capacity(path.len()-1);
            for (&i,&j) in path.iter().tuple_windows() {
                match data.travel_time.get(&(i,j)) {
                    None => {
                        trace!(i,j,pass=0,"no physical arc");
                        return None;
                    },
                    Some(&t) => tt.push(t),
                }
            }
            tt
        };

        // path is assumed to have depots included
        debug_assert_eq!(path[0], data.o_depot);
        // let start_time = max(tw_start[0], start_time);

        let mut arrival: Vec<Time> = Vec::with_capacity(path.len());
        let mut service_start: Vec<Time> = Vec::with_capacity(path.len());

        // Maps p -> index of d(p) and d -> index of p(d)
        let mut pd_map = Map::with_capacity_and_hasher(path.len(), Default::default());

        // Pass 1 (forward pass)
        arrival.push(max(tw_start[0], start_time));
        service_start.push(arrival[0]);
        pd_map.insert(data.dmap(path[0]), 0);

        for m in 1..path.len() { // index of location j in arc (i,j)


            arrival.push(service_start[m - 1] + travel_time[m - 1]);
            service_start.push(max(arrival[m], tw_start[m]));

            if arrival[m] > tw_end[m] {
                trace!(schedule=?service_start, index=m, pass=1, "time window violation");
                return None;
            }

            let j = path[m];
            // Build lookup while we're looping
            if data.is_pickup(j) {
                pd_map.insert(j+data.n, m);
            } else {
                pd_map.insert(data.pmap(j), m);
            }

            // m += 1;
        }

        let pd_map = pd_map;

        // Pass 2 (backward pass)
        // m -= 1; // index of location i
        // debug_assert_eq!(m, path.len()-1);
        let m = path.len()-1;
        let mut cum_waiting_time = service_start[m] - arrival[m]; // fix by Tang et al (Hunsaker initialised to 0)
        // travel_time[m] is the travel time from loc m to m+1;
        for m in (0..path.len()-1).rev() {
            // m -= 1;
            service_start[m] = arrival[m+1] - travel_time[m];
            let i = path[m];
            if data.is_pickup(i) || data.o_depot == i {
                debug_assert!(pd_map.contains_key(&i));
                if let Some(&m_d) = pd_map.get(&i) { // in restricted fragments we may have unmatched pickups
                    let delta = service_start[m_d] as i64 - service_start[m] as i64 - data.max_ride_time[&i] as i64;
                    if delta >= 0 {
                        if delta > cum_waiting_time as i64 {
                            trace!(schedule=?service_start, index=m, pass=2, "ride time violation");
                            return None;
                        }
                        service_start[m] += delta as Time;
                        cum_waiting_time -= delta as Time;

                        // Fix by Tang et al
                        for q in m..path.len()-1 {
                            service_start[q + 1] = max(service_start[q + 1], service_start[q] + travel_time[q])
                        }
                    }
                    if service_start[m] > tw_end[m] {
                        trace!(schedule=?service_start, index=m, pass=2, "time window violation");
                        return None;
                    }
                }
            }
            cum_waiting_time += service_start[m] - arrival[m];
        }

        // Pass 3 (forward pass #2)

        // m is index of j in arc (i,j)
        for m in 1..path.len()-1 {
            arrival[m] = service_start[m-1] + travel_time[m-1];
            service_start[m] = max(service_start[m], arrival[m]);

            if service_start[m] > tw_end[m] { return None; }

            let j = path[m];
            if data.is_delivery(j) || j == data.d_depot {
                debug_assert!(pd_map.contains_key(&j));
                if let Some(&m_p) = pd_map.get(&j) {
                    if service_start[m] - service_start[m_p] > data.max_ride_time[&data.pmap(j)] {
                        trace!(schedule=?service_start, index=m, pass=3,"ride time violation");
                        return None;
                    }
                }
            }

            // m += 1;
        }
        trace!(schedule=?service_start, "schedule found");
        return Some(service_start);
    }

    #[inline]
    pub fn try_get_early_depots(path: &[Loc], data: &DarpInstance, start_time: Time) -> Option<Vec<Time>> {
        return try_get_early(&append_depots(path, data), data, start_time)
    }


    #[inline]
    pub fn check(path: &[Loc], data: &DarpInstance, start_time: Time) -> bool {
        return try_get_early(path, data, start_time).is_some();
    }

    #[inline]
    pub fn check_depots(path: &[Loc], data: &DarpInstance, start_time: Time) -> bool {
        return check(&append_depots(path, data), data, start_time);
    }

    fn append_depots(path: &[Loc], data: &DarpInstance) -> Vec<Loc> {
        debug_assert_ne!(path[0], data.o_depot);
        debug_assert_ne!(*path.last().unwrap(), data.d_depot);
        let mut new_path = Vec::with_capacity(path.len()+2);
        new_path.push(data.o_depot);
        new_path.extend_from_slice(path);
        new_path.push(data.d_depot);
        return new_path
    }

}

