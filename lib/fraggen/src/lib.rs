#!feature(trait_alias,const_fn)
use std::path::Path;
use std::hash::Hash;
use fnv::{FnvHashMap, FnvHashSet};

pub mod sdarp;
pub mod data;
mod uid;
pub use uid::IntUid;

pub type Map<K, V> = FnvHashMap<K, V>;
pub type Set<T> = FnvHashSet<T>;


#[macro_export]
macro_rules! function_name {
    () => {{
        // Okay, this is ugly, I get it. However, this is the best we can get on a stable rust.
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        // `3` is the length of the `::f`.
        &name[..name.len() - 3]
    }};
}


mod logging_setup {
    use super::*;
    use tracing_subscriber::{EnvFilter, fmt, registry, prelude::*};
    use tracing_appender::{non_blocking, non_blocking::WorkerGuard};
    use std::fs::OpenOptions;

    fn build_and_set_global_subscriber<P>(logfile: Option<P>, is_test : bool) -> Option<WorkerGuard> where
        P : AsRef<Path>
    {
        let stderr_log = fmt::layer();
        let env_filter = EnvFilter::from_default_env();
        let r = registry().with(stderr_log).with(env_filter);

        let flush_guard = match logfile {
            Some(p) => {
                let logfile = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(p).unwrap();
                let (writer, _guard) = non_blocking::NonBlockingBuilder::default()
                    .lossy(false)
                    .finish(logfile);
                let json = fmt::layer()
                    .json()
                    .with_span_list(true)
                    .with_current_span(false)
                    .with_writer(writer);

                let r = r.with(json);
                if is_test { r.try_init().ok(); }
                else { r.init(); }
                Some(_guard)
            },
            None => {
                if is_test { r.try_init().ok(); }
                else { r.init(); }
                None
            }
        };
        return flush_guard
    }

    pub fn init_logging(logfile: Option<impl AsRef<Path>>) -> Option<WorkerGuard> {
        return build_and_set_global_subscriber(logfile, false);
    }

    #[allow(dead_code)]
    pub(crate) fn init_test_logging(logfile: Option<impl AsRef<Path>>) -> Option<WorkerGuard> {
        return build_and_set_global_subscriber(logfile, true);
    }
}
pub use logging_setup::*;



#[macro_export]
macro_rules! map (
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = Map::default();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);


pub(crate) mod utils {
    use num;
    use std::ops::{BitOrAssign, ShrAssign};

    pub struct Biterator<B> {
        bits : B,
        ones : u32,
        next_index: u32,
    }

    impl<B : num::Unsigned + num::PrimInt> Biterator<B> {
        pub fn new(val : B) -> Self {
            Self{ bits: val, ones: 0, next_index: 0 }
        }
    }

    impl<B : num::Unsigned + num::Zero + num::PrimInt + ShrAssign + From<u32>> Iterator for Biterator<B> {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            if self.ones > 0 {
                let val = self.next_index;
                self.ones -= 1;
                self.next_index += 1;
                return Some(val);
            } else if self.bits.is_zero() {
                return None;
            } else {
                let nz = self.bits.trailing_zeros();
                // debug_assert_ne!(nz, 0);
                self.bits >>= nz.into();
                self.next_index += nz;
                let no = (!self.bits).trailing_zeros();
                self.ones = no;
                self.bits >>= no.into();
                return self.next();
            }
        }
    }

    #[allow(dead_code)]
    pub fn pack_bits<T, S>(bit_indices: &[T]) -> S where
        T : num::PrimInt + num::Unsigned + Into<u8>,
        S : num::PrimInt + num::Unsigned + From<u8> + BitOrAssign,
    {
        let mut bit_indices = bit_indices.iter();
        match bit_indices.next() {
            None => return 0.into(),
            Some(i) => {
                let i : u8 = (*i).into();
                let mut val : S = (1u8 << i).into();
                for i in bit_indices {
                    let i : u8 = (*i).into();
                    val |= (1u8 << i).into();
                }
                return val;
            }
        }
    }



    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn biterate() {
            fn get_inds(val : u64) -> Vec<u32> {
                return Biterator::new(val).collect();
            }

            assert_eq!(get_inds(0), Vec::<u32>::new());
            assert_eq!(get_inds(1), vec![0u32]);
            assert_eq!(get_inds(0b010101010101), vec![0,2,4,6,8,10]);
            assert_eq!(get_inds(0b01110010011), vec![0,1,4,7,8,9]);
            assert_eq!(get_inds(0b1111111111), vec![0,1,2,3,4,5,6,7,8,9]);
        }

    }
}