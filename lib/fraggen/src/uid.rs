pub trait IntUid: Copy + Eq + Ord {
    type Raw;
    fn new() -> Self;
    fn raw(&self) -> Self::Raw;
}

#[macro_export(local_inner_macros)]
macro_rules! _impl_define_uint_id_type {
    ($name:ident, $type:ty, $atomic_type:ty) => {
        #[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
        pub struct $name($type);

        impl crate::types::uid::IntUid for $name {
            type Raw = $type;

            fn new() -> Self {
                static NEXT_ID: $atomic_type = <$atomic_type>::new(0);
                return Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
            }

            fn raw(&self) -> Self::Raw {
                return self.0
            }
        }
    }
}

#[macro_export(local_inner_macros)]
macro_rules! _impl_define_nonzero_uint_id_type {
    ($name:ident, $nonzerotype:ty, $atomic_type:ty) => {
        #[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
        pub struct $name($nonzerotype);

        impl crate::IntUid for $name {
            type Raw = $nonzerotype;

            fn new() -> Self {
                static NEXT_ID: $atomic_type = <$atomic_type>::new(1);
                let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Self(unsafe { <$nonzerotype>::new_unchecked(id) })
            }

            fn raw(&self) -> Self::Raw {
                return self.0
            }
        }
    }
}

#[macro_export]
macro_rules! define_nonzero_usize_id_type { ($name:ident) => { _impl_define_nonzero_uint_id_type!{$name, std::num::NonZeroUsize, std::sync::atomic::AtomicUsize} } }
#[macro_export]
macro_rules! define_nonzero_u64_id_type { ($name:ident) => { _impl_define_nonzero_uint_id_type!{$name, std::num::NonZeroU64, std::sync::atomic::AtomicU64} } }
#[macro_export]
macro_rules! define_nonzero_u32_id_type { ($name:ident) => { _impl_define_nonzero_uint_id_type!{$name, std::num::NonZeroU32, std::sync::atomic::AtomicU32} } }
#[macro_export]
macro_rules! define_nonzero_u16_id_type { ($name:ident) => { _impl_define_nonzero_uint_id_type!{$name, std::num::NonZeroU16, std::sync::atomic::AtomicU16} } }
#[macro_export]
macro_rules! define_nonzero_u8_id_type { ($name:ident) => { _impl_define_nonzero_uint_id_type!{$name, std::num::NonZeroU8, std::sync::atomic::AtomicU8} } }

#[macro_export]
macro_rules! define_usize_id_type { ($name:ident) => { _impl_define_uint_id_type!{$name, usize, std::sync::atomic::AtomicUsize} } }
#[macro_export]
macro_rules! define_u64_id_type { ($name:ident) => { _impl_define_uint_id_type!{$name, u64, std::sync::atomic::AtomicU64} } }
#[macro_export]
macro_rules! define_u32_id_type { ($name:ident) => { _impl_define_uint_id_type!{$name, u32, std::sync::atomic::AtomicU32} } }
#[macro_export]
macro_rules! define_u16_id_type { ($name:ident) => { _impl_define_uint_id_type!{$name, u16, std::sync::atomic::AtomicU16} } }
#[macro_export]
macro_rules! define_u8_id_type { ($name:ident) => { _impl_define_uint_id_type!{$name, u8, std::sync::atomic::AtomicU8} } }
