use crate::graph::DagreGraph;
use crate::types::NodeLabel;
use std::sync::atomic::{AtomicUsize, Ordering};

static UNIQUE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn unique_id(prefix: &str) -> String {
    let id = UNIQUE_ID_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    format!("{}{}", prefix, id)
}

pub fn reset_unique_id() {
    UNIQUE_ID_COUNTER.store(0, Ordering::SeqCst);
}

pub fn add_dummy_node(
    g: &mut DagreGraph,
    dummy_type: &str,
    mut label: NodeLabel,
    name: &str,
) -> String {
    let mut v: String;
    loop {
        v = unique_id(name);
        if !g.has_node(&v) {
            break;
        }
    }
    label.dummy = Some(dummy_type.to_string());
    g.set_node(&v, Some(label));
    v
}

pub fn as_non_compound_graph(g: &DagreGraph) -> DagreGraph {
    // OPTIMIZATION: Direct clone is much faster than set_node_with_shared_data
    let mut graph = g.clone();

    // Remove compound-specific fields
    graph.compound = false;
    graph.parent = None;
    graph.children = None;

    graph
}

pub fn max_rank(g: &DagreGraph) -> Option<i32> {
    g.nodes
        .values()
        .filter_map(|node| {
            let label = node.label.read();
            let rank = label.rank;
            drop(label);
            rank
        })
        .max()
}

pub fn build_layer_matrix(g: &DagreGraph) -> Vec<Vec<std::sync::Arc<str>>> {
    let max_rank_value = max_rank(g).unwrap_or(0);
    let length = (max_rank_value + 1) as usize;
    let mut layering: Vec<Vec<(usize, std::sync::Arc<str>)>> = vec![Vec::new(); length];

    // Collect nodes with rank and order
    for node in g.nodes.values() {
        let label = node.label.read();
        if let Some(rank) = label.rank {
            drop(label); // Release lock early
                         // Lock-free atomic read for order
            let order = node.order_atomic.load(std::sync::atomic::Ordering::Relaxed);
            if (rank as usize) < length {
                layering[rank as usize].push((order, node.v.clone()));
            }
        }
    }

    // Sort each layer by order and extract node IDs
    let result: Vec<Vec<std::sync::Arc<str>>> = layering
        .into_iter()
        .map(|mut layer| {
            layer.sort_unstable_by_key(|(order, _)| *order);
            layer.into_iter().map(|(_, v)| v).collect()
        })
        .filter(|layer: &Vec<std::sync::Arc<str>>| !layer.is_empty())
        .collect();

    result
}

// Note: This function is not used - kept for potential future use
#[allow(dead_code)]
pub(crate) fn flatten<T: Clone>(list: Vec<Vec<T>>) -> Vec<T> {
    list.into_iter().flatten().collect()
}
