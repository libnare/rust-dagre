use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// Constants for rank directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankDirection {
    #[serde(rename = "TB")]
    TB, // Top to Bottom
    #[serde(rename = "BT")]
    BT, // Bottom to Top
    #[serde(rename = "LR")]
    LR, // Left to Right
    #[serde(rename = "RL")]
    RL, // Right to Left
}

impl RankDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            RankDirection::TB => "TB",
            RankDirection::BT => "BT",
            RankDirection::LR => "LR",
            RankDirection::RL => "RL",
        }
    }
}

impl Default for RankDirection {
    fn default() -> Self {
        RankDirection::TB
    }
}

// Constants for label positions (internal use only)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LabelPosition {
    #[serde(rename = "l")]
    Left,
    #[serde(rename = "r")]
    Right,
    #[serde(rename = "c")]
    Center,
}

impl Default for LabelPosition {
    fn default() -> Self {
        LabelPosition::Right
    }
}

// Constants for ranker algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ranker {
    #[serde(rename = "network-simplex")]
    NetworkSimplex,
    #[serde(rename = "tight-tree")]
    TightTree,
    #[serde(rename = "longest-path")]
    LongestPath,
}

impl Default for Ranker {
    fn default() -> Self {
        Ranker::NetworkSimplex
    }
}

// Constants for alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Alignment {
    #[serde(rename = "ul")]
    UL, // Up-Left
    #[serde(rename = "ur")]
    UR, // Up-Right
    #[serde(rename = "dl")]
    DL, // Down-Left
    #[serde(rename = "dr")]
    DR, // Down-Right
}

// Point for edge routing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// ===== Internal types (order optimization) =====

// Barycenter entry for order optimization
#[derive(Debug, Clone)]
pub(crate) struct BarycenterEntry {
    pub(crate) v: std::sync::Arc<str>,
    pub(crate) barycenter: Option<f64>,
    pub(crate) weight: Option<f64>,
}

// Mapped entry for conflict resolution
#[derive(Debug, Clone)]
pub(crate) struct MappedEntry {
    pub(crate) indegree: usize,
    pub(crate) in_entries: Vec<usize>, // Indexes into the entries array
    pub(crate) out_entries: Vec<usize>, // Indexes into the entries array
    pub(crate) vs: Vec<Arc<str>>,
    pub(crate) i: usize,
    pub(crate) barycenter: Option<f64>,
    pub(crate) weight: Option<f64>,
    pub(crate) merged: bool,
}

// Sort result
#[derive(Debug, Clone)]
pub(crate) struct SortResult {
    pub(crate) vs: Vec<Arc<str>>,
    pub(crate) barycenter: Option<f64>,
    pub(crate) weight: Option<f64>,
}

// Node in the graph layout
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub v: String,
    pub width: f64,
    pub height: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
}

// Edge in the graph layout
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub v: String,
    pub w: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minlen: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labeloffset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labelpos: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<Point>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
}

// Configuration options for the layout algorithm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranksep: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edgesep: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodesep: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rankdir: Option<RankDirection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<Ranker>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub align: Option<Alignment>,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        LayoutConfig {
            ranksep: Some(50.0),
            edgesep: Some(20.0),
            nodesep: Some(50.0),
            rankdir: Some(RankDirection::TB),
            ranker: Some(Ranker::NetworkSimplex),
            align: None,
        }
    }
}

// Internal state maintained during layout computation
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutState {
    pub width: f64,
    pub height: f64,
    pub node_rank_factor: Option<f64>,
    pub nesting_root: Option<std::sync::Arc<str>>,
    pub max_rank: Option<i32>,
    pub dummy_chains: Option<Vec<std::sync::Arc<str>>>,
    pub log: Option<String>,
}

impl Default for LayoutState {
    fn default() -> Self {
        LayoutState {
            width: 0.0,
            height: 0.0,
            node_rank_factor: None,
            nesting_root: None,
            max_rank: None,
            dummy_chains: None,
            log: None,
        }
    }
}

// ===== Internal types (layout information) =====
// These are exposed for advanced users who want to manipulate DagreGraph directly

/// Internal node label with layout information
///
/// **Advanced API**: Most users should not need to use this directly.
/// Use the `layout()` function instead.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeLabel {
    pub width: f64,
    pub height: f64,
    pub rank: Option<i32>,
    pub order: Option<usize>,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub dummy: Option<String>,
    pub self_edges: Option<Vec<SelfEdge>>,
    pub border_top: Option<std::sync::Arc<str>>,
    pub border_bottom: Option<std::sync::Arc<str>>,
    pub border_left: Option<Vec<std::sync::Arc<str>>>,
    pub border_right: Option<Vec<std::sync::Arc<str>>>,
    pub min_rank: Option<i32>,
    pub max_rank: Option<i32>,
    pub parent: Option<std::sync::Arc<str>>,
    pub low: Option<i32>,
    pub lim: Option<i32>,
    pub border_type: Option<String>,
    pub edge_label: Option<Box<EdgeLabel>>,
    pub edge_obj: Option<EdgeKey>,
    pub label: Option<Box<EdgeLabel>>,
    pub labelpos: Option<String>,
    pub labeloffset: Option<f64>,
    pub label_rank: Option<i32>,
    pub forward_name: Option<String>,
    pub reversed: Option<bool>,
    pub points: Option<Vec<Point>>,
    pub weight: Option<i32>,
    pub minlen: Option<i32>,
    pub cutvalue: Option<i32>,
    pub e: Option<EdgeKey>,
}

impl NodeLabel {
    pub fn new(width: f64, height: f64) -> Self {
        NodeLabel {
            width,
            height,
            rank: None,
            order: None,
            x: None,
            y: None,
            dummy: None,
            self_edges: None,
            border_top: None,
            border_bottom: None,
            border_left: None,
            border_right: None,
            min_rank: None,
            max_rank: None,
            parent: None,
            low: None,
            lim: None,
            border_type: None,
            edge_label: None,
            edge_obj: None,
            label: None,
            labelpos: None,
            labeloffset: None,
            label_rank: None,
            forward_name: None,
            reversed: None,
            points: None,
            weight: None,
            minlen: None,
            cutvalue: None,
            e: None,
        }
    }
}

impl Default for NodeLabel {
    fn default() -> Self {
        NodeLabel::new(0.0, 0.0)
    }
}

/// Internal edge label with layout information
///
/// **Advanced API**: Most users should not need to use this directly.
/// Use the `layout()` function instead.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLabel {
    pub weight: f64,
    pub minlen: i32,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub labeloffset: Option<f64>,
    pub labelpos: Option<String>,
    pub points: Option<Vec<Point>>,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub dummy: Option<String>,
    pub forward_name: Option<String>,
    pub reversed: Option<bool>,
    pub label_rank: Option<i32>,
    pub nesting_edge: Option<bool>,
    pub cutvalue: Option<f64>,
}

impl EdgeLabel {
    pub fn new(weight: f64, minlen: i32) -> Self {
        EdgeLabel {
            weight,
            minlen,
            width: None,
            height: None,
            labeloffset: None,
            labelpos: None,
            points: None,
            x: None,
            y: None,
            dummy: None,
            forward_name: None,
            reversed: None,
            label_rank: None,
            nesting_edge: None,
            cutvalue: None,
        }
    }
}

impl Default for EdgeLabel {
    fn default() -> Self {
        EdgeLabel::new(1.0, 1)
    }
}

// Self edge structure (internal only)
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SelfEdge {
    pub(crate) e: EdgeKey,
    pub(crate) label: EdgeLabel,
}

/// Edge key for identifying edges
///
/// **Advanced API**: Most users should not need to use this directly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgeKey {
    pub v: Arc<str>,
    pub w: Arc<str>,
    pub name: Option<Arc<str>>,
}

impl EdgeKey {
    pub(crate) fn new(v: Arc<str>, w: Arc<str>, name: Option<Arc<str>>) -> Self {
        EdgeKey { v, w, name }
    }
}

// ===== Test/Debug types (JSON serialization) =====

// Graph data structure for JSON input
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

// Layout result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LayoutResult {
    pub width: f64,
    pub height: f64,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
