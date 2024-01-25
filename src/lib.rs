#![feature(adt_const_params)]
#![feature(const_option)]
#![feature(const_float_bits_conv)]
#![feature(const_fn_floating_point_arithmetic)]

pub mod ball;
pub mod grid_loose_quadtree;
pub mod quadtree;
pub mod rect;
pub mod up_search_quadtree;

#[cfg(test)]
pub mod test;

pub use grid_loose_quadtree::*;
pub use quadtree::*;
pub use up_search_quadtree::*;

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
pub struct Looseness(u64);

impl Looseness {
    /// The given [`n`] must be in 1.0..=2.0
    pub const fn from_f64(n: f64) -> Option<Looseness> {
        if n >= 1.0 && n <= 2.0 {
            Some(Looseness(n.to_bits()))
        } else {
            None
        }
    }

    pub const fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}
