#![doc = include_str!("../README.md")]

sidex::include_bundle! {
    /// Data structures generated by Sidex.
    momba_model as generated
}

pub use generated::*;
