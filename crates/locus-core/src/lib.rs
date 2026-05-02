//! Core types for the Locus control-systems environment.
//!
//! This crate has no heavy numerical dependencies. It defines polynomials,
//! transfer functions, and pure algebraic system connections. Eigendecomp,
//! root-finding, and matrix-based representations live in `locus-num`.

mod connection;
mod error;
mod polynomial;
mod tf;

pub use connection::{feedback, feedback_positive, parallel, series};
pub use error::CoreError;
pub use polynomial::{poly_div, Polynomial};
pub use tf::TransferFunction;

/// Continuous-time scalar; seconds throughout the workspace.
pub type Time = f64;
