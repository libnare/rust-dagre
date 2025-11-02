// Rust port of dagre.js - a directed graph layout library
// Based on: https://github.com/dagrejs/dagre

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

// ===== Essential Public API (for 99% of users) =====
/// Main layout function - computes positions for all nodes and edges
pub use layout::layout;

/// Input/output types for the layout algorithm
pub use types::{Node, Edge, LayoutConfig, LayoutState, Point};

/// Configuration enums
pub use types::{RankDirection, Ranker, Alignment};

// ===== Advanced Public API (for direct graph manipulation) =====
/// Graph data structure for advanced use cases
pub use graph::DagreGraph;

/// Advanced types for direct graph manipulation
/// 
/// **Note**: These types are only needed if you're directly manipulating `DagreGraph`.
/// Most users should just use the `layout()` function.
pub use types::{NodeLabel, EdgeLabel, EdgeKey};

// ===== Utility functions =====
/// Generate unique IDs and reset counter (useful for testing)
pub use utils::{unique_id, reset_unique_id};
