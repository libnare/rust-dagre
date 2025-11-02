use crate::acyclic::{acyclic_run, acyclic_undo};
use crate::coordinate_system::{coordinate_system_adjust, coordinate_system_undo, translate_graph};
use crate::edges::{
    assign_node_intersects, fixup_edge_label_coords, insert_self_edges, make_space_for_edge_labels,
    position_self_edges, remove_self_edges,
};
use crate::graph::DagreGraph;
use crate::nesting::{
    add_border_segments, assign_rank_min_max, nesting_graph_cleanup, nesting_graph_run,
    parent_dummy_chains, remove_border_nodes,
};
use crate::normalize::{
    denormalize, inject_edge_label_proxies, normalize, remove_edge_label_proxies,
    remove_empty_ranks,
};
use crate::order::order;
use crate::position::position;
use crate::rank::rank;
use crate::types::{Edge, EdgeLabel, LayoutConfig, LayoutState, Node, NodeLabel};
use crate::utils::reset_unique_id;

pub fn layout(
    nodes: &mut [Node],
    edges: &mut [Edge],
    layout_config: &LayoutConfig,
    state: &mut LayoutState,
) {
    let total_start = std::time::Instant::now();
    reset_unique_id();

    let mut g = DagreGraph::new(true, true);

    for node in nodes.iter() {
        let label = NodeLabel::new(node.width, node.height);
        g.set_node(&node.v, Some(label));

        if let Some(ref parent) = node.parent {
            if !parent.is_empty() {
                g.set_parent(&node.v, Some(parent));
            }
        }
    }

    for edge in edges.iter() {
        let label = EdgeLabel {
            minlen: edge.minlen.unwrap_or(1),
            weight: edge.weight.unwrap_or(1.0),
            width: edge.width,
            height: edge.height,
            labeloffset: edge.labeloffset,
            labelpos: edge.labelpos.clone(),
            points: None,
            x: None,
            y: None,
            dummy: None,
            forward_name: None,
            reversed: None,
            label_rank: None,
            nesting_edge: None,
            cutvalue: None,
        };
        g.set_edge(&edge.v, &edge.w, Some(label), None);
    }

    let mut layout = layout_config.clone();
    if layout.ranksep.is_none() {
        layout.ranksep = Some(50.0);
    }
    if layout.edgesep.is_none() {
        layout.edgesep = Some(20.0);
    }
    if layout.nodesep.is_none() {
        layout.nodesep = Some(50.0);
    }

    let mut step_start = std::time::Instant::now();
    make_space_for_edge_labels(&mut g, state, &mut layout);
    remove_self_edges(&mut g);
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] make_space + remove_self_edges: {:?}",
        step_start.elapsed()
    );

    step_start = std::time::Instant::now();
    acyclic_run(&mut g);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] acyclic_run: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    nesting_graph_run(&mut g, state);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] nesting_graph_run: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    rank(&mut g, &layout);
    let max_rank_after_rank = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .max()
        .unwrap_or(0);
    let min_rank_after_rank = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .min()
        .unwrap_or(0);
    let unique_ranks = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .collect::<std::collections::HashSet<_>>()
        .len();
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] rank: {:?}, min={}, max={}, unique={}",
        step_start.elapsed(),
        min_rank_after_rank,
        max_rank_after_rank,
        unique_ranks
    );

    step_start = std::time::Instant::now();
    inject_edge_label_proxies(&mut g);

    let unique_after_inject = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .collect::<std::collections::HashSet<_>>()
        .len();
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] after inject_edge_label_proxies: unique={}",
        unique_after_inject
    );

    remove_empty_ranks(&mut g, state);

    let unique_after_remove_empty = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .collect::<std::collections::HashSet<_>>()
        .len();
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] after remove_empty_ranks: unique={}",
        unique_after_remove_empty
    );

    nesting_graph_cleanup(&mut g, state);
    assign_rank_min_max(&mut g, state);

    let max_rank_before = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .max()
        .unwrap_or(0);
    let unique_ranks_before = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .collect::<std::collections::HashSet<_>>()
        .len();
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] before normalize: max={}, unique={}",
        max_rank_before, unique_ranks_before
    );

    remove_edge_label_proxies(&mut g);
    normalize(&mut g, state);

    let max_rank_after = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .max()
        .unwrap_or(0);
    let unique_ranks_after = g
        .nodes
        .values()
        .filter_map(|n| n.label.read().rank)
        .collect::<std::collections::HashSet<_>>()
        .len();
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] after normalize: max={}, unique={}",
        max_rank_after, unique_ranks_after
    );
    #[cfg(debug_assertions)]
    eprintln!("[PERF] normalize pipeline: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    parent_dummy_chains(&mut g);
    add_border_segments(&mut g);
    #[cfg(debug_assertions)]
    eprintln!(
        "[PERF] parent_dummy_chains + add_border_segments: {:?}",
        step_start.elapsed()
    );

    step_start = std::time::Instant::now();
    order(&mut g, &layout);
    eprintln!("[PERF] order: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    insert_self_edges(&mut g);
    eprintln!("[PERF] insert_self_edges: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    coordinate_system_adjust(&mut g, &layout);
    eprintln!(
        "[PERF] coordinate_system_adjust: {:?}",
        step_start.elapsed()
    );

    step_start = std::time::Instant::now();
    position(&mut g, &layout);
    eprintln!("[PERF] position: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    position_self_edges(&mut g);
    eprintln!("[PERF] position_self_edges: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    remove_border_nodes(&mut g);
    eprintln!("[PERF] remove_border_nodes: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    denormalize(&mut g, state);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] denormalize: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    fixup_edge_label_coords(&mut g);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] fixup_edge_label_coords: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    coordinate_system_undo(&mut g, &layout);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] coordinate_system_undo: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    translate_graph(&mut g, state);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] translate_graph: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    assign_node_intersects(&mut g);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] assign_node_intersects: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    acyclic_undo(&mut g);
    #[cfg(debug_assertions)]
    eprintln!("[PERF] acyclic_undo: {:?}", step_start.elapsed());

    for node in nodes.iter_mut() {
        if let Some(graph_node) = g.node(&node.v) {
            node.x = graph_node.label.read().x;
            node.y = graph_node.label.read().y;

            if g.has_children(Some(&node.v)) {
                node.width = graph_node.label.read().width;
                node.height = graph_node.label.read().height;
            }
        }
    }

    for edge in edges.iter_mut() {
        if let Some(graph_edge) = g.edge(&edge.v, &edge.w) {
            edge.points = graph_edge.label.points.clone();
            edge.x = graph_edge.label.x;
            edge.y = graph_edge.label.y;
        }
    }

    #[cfg(debug_assertions)]
    eprintln!("[PERF] TOTAL layout time: {:?}", total_start.elapsed());
}
