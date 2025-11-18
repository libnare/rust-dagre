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

    make_space_for_edge_labels(&mut g, state, &mut layout);
    remove_self_edges(&mut g);

    acyclic_run(&mut g);

    nesting_graph_run(&mut g, state);

    rank(&mut g, &layout);

    inject_edge_label_proxies(&mut g);

    remove_empty_ranks(&mut g, state);

    nesting_graph_cleanup(&mut g, state);
    assign_rank_min_max(&mut g, state);

    remove_edge_label_proxies(&mut g);
    normalize(&mut g, state);

    parent_dummy_chains(&mut g);
    add_border_segments(&mut g);

    order(&mut g, &layout);

    insert_self_edges(&mut g);

    coordinate_system_adjust(&mut g, &layout);

    position(&mut g, &layout);

    position_self_edges(&mut g);

    remove_border_nodes(&mut g);

    denormalize(&mut g, state);

    fixup_edge_label_coords(&mut g);

    coordinate_system_undo(&mut g, &layout);

    translate_graph(&mut g, state);

    assign_node_intersects(&mut g);

    acyclic_undo(&mut g);

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
}
