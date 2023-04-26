use std::{
    alloc,
    ptr::NonNull,
    sync::atomic::{self, AtomicUsize},
};

use parking_lot::Mutex;

pub struct StateStore {}

pub struct TransitionsStore {}

pub struct DestinationsStore {}
