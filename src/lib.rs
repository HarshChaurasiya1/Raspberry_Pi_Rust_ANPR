//! Raspberry Pi 5 High-Performance Grammar-Constrained ANPR Library in Rust.

pub mod config;
pub mod camera;
pub mod image;
pub mod detector;
pub mod ocr;
pub mod grammar;
pub mod pipeline;
pub mod benchmark;
pub mod tracker;

pub use config::Config;
pub use pipeline::{ANPRPipeline, PlateResult};
pub use detector::Detection;
pub use tracker::{PlateTracker, TrackedPlate};
