use crate::graph::DagreGraph;
use crate::types::{EdgeLabel, LayoutState, NodeLabel};
use crate::utils::add_dummy_node;
use ahash::AHashMap as HashMap;

pub fn nesting_graph_run(g: &mut DagreGraph, state: &mut LayoutState) {
    let root = add_dummy_node(g, "root", NodeLabel::new(0.0, 0.0), "_root");

    let depths = tree_depths(g);
    let height = depths.values().max().copied().unwrap_or(0) - 1;
    let node_sep = 2 * height + 1;

    state.nesting_root = Some(crate::graph::arc_str(&root));

    let edges_to_update: Vec<(String, String, Option<String>, i32)> = g
        .edges
        .values()
        .map(|e| {
            (
                e.v.as_ref().to_string(),
                e.w.as_ref().to_string(),
                e.name.as_ref().map(|s| s.as_ref().to_string()),
                e.label.minlen,
            )
        })
        .collect();

    for (v, w, _name, minlen) in edges_to_update {
        if let Some(edge) = g.edge_mut(&v, &w) {
            edge.label.minlen = minlen * node_sep;
        }
    }

    let total_weight: f64 = g.edges.values().map(|e| e.label.weight).sum();
    let weight = total_weight + 1.0;

    let children: Vec<std::sync::Arc<str>> = g.children(None).collect();
    for child in children {
        dfs(g, &root, node_sep, weight, height, &depths, &child);
    }

    state.node_rank_factor = Some(node_sep as f64);
}

fn tree_depths(g: &DagreGraph) -> HashMap<std::sync::Arc<str>, i32> {
    let mut depths = HashMap::new();

    fn dfs_depth(
        g: &DagreGraph,
        v: &str,
        depth: i32,
        depths: &mut HashMap<std::sync::Arc<str>, i32>,
    ) {
        let children: Vec<std::sync::Arc<str>> = g.children(Some(v)).collect();
        for child in children {
            dfs_depth(g, &child, depth + 1, depths);
        }
        depths.insert(crate::graph::arc_str(v), depth);
    }

    let root_children: Vec<std::sync::Arc<str>> = g.children(None).collect();
    for v in root_children {
        dfs_depth(g, &v, 1, &mut depths);
    }

    depths
}

fn dfs(
    g: &mut DagreGraph,
    root: &str,
    node_sep: i32,
    weight: f64,
    height: i32,
    depths: &HashMap<std::sync::Arc<str>, i32>,
    v: &str,
) {
    let children: Vec<std::sync::Arc<str>> = g.children(Some(v)).collect();

    if children.is_empty() {
        if v != root {
            let mut label = EdgeLabel::new(0.0, node_sep);
            label.nesting_edge = Some(true);
            g.set_edge(root, v, Some(label), None);
        }
        return;
    }

    let top = add_dummy_node(g, "border", NodeLabel::new(0.0, 0.0), "_bt");
    let bottom = add_dummy_node(g, "border", NodeLabel::new(0.0, 0.0), "_bb");

    g.has_border = Some(true);
    g.set_parent(&top, Some(v));
    g.set_parent(&bottom, Some(v));

    if let Some(node) = g.node_mut(v) {
        node.label.write().border_top = Some(crate::graph::arc_str(&top));
        node.label.write().border_bottom = Some(crate::graph::arc_str(&bottom));
    }

    for child in children {
        dfs(g, root, node_sep, weight, height, depths, &child);

        let (child_top, child_bottom, has_border) = if let Some(child_node) = g.node(&child) {
            let child_top = child_node
                .label
                .read()
                .border_top
                .clone()
                .unwrap_or_else(|| child.clone());
            let child_bottom = child_node
                .label
                .read()
                .border_bottom
                .clone()
                .unwrap_or_else(|| child.clone());
            let has_border = child_node.label.read().border_top.is_some();
            (child_top, child_bottom, has_border)
        } else {
            continue;
        };

        let this_weight: f64 = if has_border { weight } else { 2.0 * weight };
        let minlen = if child_top == child_bottom {
            height - depths.get(v).copied().unwrap_or(0) + 1
        } else {
            1
        };

        let mut label_top = EdgeLabel::new(this_weight, minlen);
        label_top.nesting_edge = Some(true);
        g.set_edge(&top, &child_top, Some(label_top), None);

        let mut label_bottom = EdgeLabel::new(this_weight, minlen);
        label_bottom.nesting_edge = Some(true);
        g.set_edge(&child_bottom, &bottom, Some(label_bottom), None);
    }

    if g.parent(v).is_none() {
        let mut label = EdgeLabel::new(0.0, height + depths.get(v).copied().unwrap_or(0));
        label.nesting_edge = Some(true);
        g.set_edge(root, &top, Some(label), None);
    }
}

pub fn nesting_graph_cleanup(g: &mut DagreGraph, state: &mut LayoutState) {
    if let Some(root) = &state.nesting_root {
        g.remove_node(root);
        state.nesting_root = None;
    }

    let edges_to_remove: Vec<(String, String, Option<String>)> = g
        .edges
        .values()
        .filter(|e| e.label.nesting_edge == Some(true))
        .map(|e| {
            (
                e.v.as_ref().to_string(),
                e.w.as_ref().to_string(),
                e.name.as_ref().map(|s| s.as_ref().to_string()),
            )
        })
        .collect();

    for (v, w, name) in edges_to_remove {
        g.remove_edge(&v, &w, name.as_deref());
    }
}

pub fn assign_rank_min_max(g: &mut DagreGraph, state: &mut LayoutState) {
    let mut min = i32::MAX;

    for node in g.nodes.values() {
        if let Some(rank) = node.label.read().rank {
            min = min.min(rank);
        }
    }

    for node in g.nodes.values_mut() {
        let mut label = node.label.write();
        if let Some(rank) = label.rank {
            label.rank = Some(rank - min);
        }
    }

    let mut max_rank = 0;

    if g.has_border == Some(true) {
        let nodes_with_borders: Vec<(String, std::sync::Arc<str>, std::sync::Arc<str>)> = g
            .nodes
            .values()
            .filter_map(|node| {
                if let (Some(ref top), Some(ref bottom)) = (
                    &node.label.read().border_top,
                    &node.label.read().border_bottom,
                ) {
                    Some((node.v.as_ref().to_string(), top.clone(), bottom.clone()))
                } else {
                    None
                }
            })
            .collect();

        for (v, top, bottom) in nodes_with_borders {
            if let (Some(top_node), Some(bottom_node)) = (g.node(&top), g.node(&bottom)) {
                let min_rank = top_node.label.read().rank;
                let max_rank_val = bottom_node.label.read().rank;

                if let Some(node) = g.node_mut(&v) {
                    node.label.write().min_rank = min_rank;
                    node.label.write().max_rank = max_rank_val;

                    if let Some(mr) = max_rank_val {
                        max_rank = max_rank.max(mr);
                    }
                }
            }
        }
    }

    state.max_rank = Some(max_rank);
}

pub fn parent_dummy_chains(g: &mut DagreGraph) {
    let mut postorder_nums: HashMap<std::sync::Arc<str>, (i32, i32)> = HashMap::new();
    let mut postorder_num = 0;

    fn dfs(
        g: &DagreGraph,
        v: &str,
        postorder_nums: &mut HashMap<std::sync::Arc<str>, (i32, i32)>,
        postorder_num: &mut i32,
    ) {
        let children: Vec<std::sync::Arc<str>> = g.children(Some(v)).collect();
        for child in children {
            dfs(g, &child, postorder_nums, postorder_num);
        }
        postorder_nums.insert(crate::graph::arc_str(v), (*postorder_num, *postorder_num));
        *postorder_num += 1;
    }

    let root_children: Vec<std::sync::Arc<str>> = g.children(None).collect();
    for v in root_children {
        dfs(g, &v, &mut postorder_nums, &mut postorder_num);
    }

    let edges_to_update: Vec<(String, String, Option<String>)> = g
        .edges
        .values()
        .filter_map(|edge| {
            let v = &edge.v;
            let w = &edge.w;

            if let (Some(v_node), Some(w_node)) = (g.node(v), g.node(w)) {
                let v_rank = v_node.label.read().rank;
                let w_rank = w_node.label.read().rank;

                if let (Some(vr), Some(wr)) = (v_rank, w_rank) {
                    if vr == wr {
                        if let (Some(&(v_low, _)), Some(&(w_low, _))) = (
                            postorder_nums.get(v.as_ref()),
                            postorder_nums.get(w.as_ref()),
                        ) {
                            if v_low < w_low {
                                return Some((
                                    v.as_ref().to_string(),
                                    w.as_ref().to_string(),
                                    edge.name.as_ref().map(|s| s.as_ref().to_string()),
                                ));
                            }
                        }
                    }
                }
            }
            None
        })
        .collect();

    for (v, w, _name) in edges_to_update {
        if let Some(edge) = g.edge_mut(&v, &w) {
            edge.label.minlen = 2;
        }
    }
}

pub fn add_border_segments(_g: &mut DagreGraph) {
    // This function is simplified for now
    // The TypeScript version adds border segments for compound nodes
}

pub fn remove_border_nodes(g: &mut DagreGraph) {
    let nodes_to_remove: Vec<String> = g
        .nodes
        .values()
        .filter(|node| {
            node.label.read().dummy.as_deref() == Some("border")
                || node.label.read().border_type.is_some()
        })
        .map(|node| node.v.as_ref().to_string())
        .collect();

    for v in nodes_to_remove {
        g.remove_node(&v);
    }
}
