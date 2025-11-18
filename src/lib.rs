// Rust port of dagre.js - a directed graph layout library
// Based on: https://github.com/dagrejs/dagre

#![deny(clippy::all)]

// Internal modules (implementation details)
mod acyclic;
mod coordinate_system;
mod edges;
mod nesting;
mod normalize;
mod order;
mod position;
mod rank;
mod utils;

// Public modules (user-facing API)
pub mod graph;
pub mod layout;
pub mod types;

// N-API bindings (exposed to Node.js and WASM)
pub mod napi_interface;

// ===== Essential Public API (for 99% of users) =====
/// Main layout function - computes positions for all nodes and edges
pub use layout::layout;

/// Input/output types for the layout algorithm
pub use types::{Edge, LayoutConfig, LayoutState, Node, Point};

/// Configuration enums
pub use types::{Alignment, RankDirection, Ranker};

// ===== Advanced Public API (for direct graph manipulation) =====
/// Graph data structure for advanced use cases
pub use graph::DagreGraph;

/// Advanced types for direct graph manipulation
///
/// **Note**: These types are only needed if you're directly manipulating `DagreGraph`.
/// Most users should just use the `layout()` function.
pub use types::{EdgeKey, EdgeLabel, NodeLabel};

// ===== Utility functions =====
/// Generate unique IDs and reset counter (useful for testing)
pub use utils::{reset_unique_id, unique_id};
