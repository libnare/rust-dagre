use crate::graph::DagreGraph;
use crate::types::{EdgeLabel, LayoutConfig, NodeLabel};
use crate::utils::as_non_compound_graph;
use ahash::AHashSet as HashSet;

fn slack(g: &DagreGraph, edge_v: &str, edge_w: &str) -> i32 {
    let v_node = g.node(edge_v).unwrap();
    let w_node = g.node(edge_w).unwrap();
    let edge = g.edge(edge_v, edge_w).unwrap();
    w_node.label.read().rank.unwrap() - v_node.label.read().rank.unwrap() - edge.label.minlen
}

pub fn rank(g: &mut DagreGraph, _layout: &LayoutConfig) {
    let mut temp_g = as_non_compound_graph(g);
    network_simplex(&mut temp_g);
    // No need to copy ranks back - they're shared via Rc!
}

fn longest_path(g: &mut DagreGraph) {
    let node_count = g.nodes.len();
    let mut visited = HashSet::with_capacity(node_count);

    let source_nodes: Vec<std::sync::Arc<str>> = g
        .nodes
        .values()
        .filter(|node| node.in_edges.is_empty())
        .map(|node| node.v.clone())
        .collect();

    // If no source nodes, assign rank 0 to all nodes
    if source_nodes.is_empty() {
        for node in g.nodes.values_mut() {
            node.label.write().rank = Some(0);
        }
        return;
    }

    enum StackItem {
        NodeArray(Vec<std::sync::Arc<str>>),
        Node(std::sync::Arc<str>),
    }

    let mut stack: Vec<StackItem> = vec![StackItem::NodeArray(source_nodes)];
    let mut iterations = 0;
    let max_iterations = g.nodes.len() * g.nodes.len();

    while let Some(item) = stack.pop() {
        iterations += 1;
        if iterations > max_iterations {
            // Fallback: assign sequential ranks
            for (i, node) in g.nodes.values_mut().enumerate() {
                if node.label.read().rank.is_none() {
                    node.label.write().rank = Some(i as i32);
                }
            }
            break;
        }

        match item {
            StackItem::NodeArray(mut nodes) => {
                if let Some(v) = nodes.pop() {
                    if !nodes.is_empty() {
                        stack.push(StackItem::NodeArray(nodes));
                    }

                    if !visited.contains(v.as_ref()) {
                        visited.insert(v.as_ref().to_string());

                        let children: Vec<std::sync::Arc<str>> = if let Some(node) = g.node(&v) {
                            node.out_edges
                                .iter()
                                .filter_map(|key| g.edge_by_key(key))
                                .map(|e| e.w.clone())
                                .collect()
                        } else {
                            Vec::new()
                        };

                        if !children.is_empty() {
                            stack.push(StackItem::Node(v));
                            stack.push(StackItem::NodeArray(children));
                        } else {
                            if let Some(node) = g.node_mut(&v) {
                                node.label.write().rank = Some(0);
                            }
                        }
                    }
                }
            }
            StackItem::Node(v) => {
                let mut rank = i32::MAX;

                if let Some(node) = g.node(&v) {
                    for edge_key in node.out_edges.iter() {
                        if let Some(edge) = g.edge_by_key(edge_key) {
                            if let Some(w_node) = g.node(&edge.w) {
                                if let Some(w_rank) = w_node.label.read().rank {
                                    rank = rank.min(w_rank - edge.label.minlen);
                                }
                            }
                        }
                    }
                }

                if rank == i32::MAX {
                    rank = 0;
                }

                if let Some(node) = g.node_mut(&v) {
                    node.label.write().rank = Some(rank);
                }
            }
        }
    }

    // Ensure all nodes have a rank
    for node in g.nodes.values_mut() {
        if node.label.read().rank.is_none() {
            node.label.write().rank = Some(0);
        }
    }
}
fn tight_tree(t: &mut DagreGraph, g: &DagreGraph) -> usize {
    // Reverse the nodes like TypeScript: Array.from(t.nodes.keys()).reverse()
    let mut stack: Vec<std::sync::Arc<str>> = t.nodes.keys().cloned().collect();
    stack.reverse();

    while let Some(v) = stack.pop() {
        if let Some(node) = g.node(&v) {
            for edge_key in node.in_edges.iter().chain(node.out_edges.iter()) {
                if let Some(edge) = g.edge_by_key(edge_key) {
                    let w = if v.as_ref() == edge.v.as_ref() {
                        &edge.w
                    } else {
                        &edge.v
                    };

                    if !t.has_node(w) && slack(g, &edge.v, &edge.w) == 0 {
                        t.set_node(w, Some(NodeLabel::new(0.0, 0.0)));
                        t.set_edge(&v, w, Some(EdgeLabel::new(1.0, 1)), None);
                        stack.push(w.clone());
                    }
                }
            }
        }
    }

    t.nodes.len()
}

fn network_simplex(g: &mut DagreGraph) {
    let mut simplified_g = simplify_shared(g);
    longest_path(&mut simplified_g);
    let mut t = feasible_tree_ns(&mut simplified_g);

    if let Some(root) = t.nodes.keys().next().cloned() {
        init_low_lim_values(&mut t, &root);
        init_cut_values(&mut t, &simplified_g);

        let mut iterations = 0;
        let max_iterations = simplified_g.edges.len() * 4;

        while let Some((leave_v, leave_w)) = leave_edge(&t) {
            iterations += 1;

            if iterations > max_iterations {
                break;
            }

            if let Some((enter_v, enter_w)) = enter_edge(&t, &simplified_g, &leave_v, &leave_w) {
                exchange_edges(
                    &mut t,
                    &mut simplified_g,
                    &leave_v,
                    &leave_w,
                    &enter_v,
                    &enter_w,
                );
            } else {
                break;
            }
        }
    }
}

fn simplify_shared(g: &DagreGraph) -> DagreGraph {
    // Create new graph that shares NodeLabel Rc with original (like TypeScript)
    let mut graph = DagreGraph::new(true, false);

    // Share NodeLabel Rc by cloning the Rc (not the NodeLabel)
    for node in g.nodes.values() {
        graph.set_node_with_label_rc(&node.v, node.label.clone());
    }

    // Aggregate multi-edges
    for edge in g.edges.values() {
        let simple_label = if let Some(simple_edge) = graph.edge(&edge.v, &edge.w) {
            EdgeLabel {
                weight: simple_edge.label.weight + edge.label.weight,
                minlen: simple_edge.label.minlen.max(edge.label.minlen),
                ..simple_edge.label.clone()
            }
        } else {
            EdgeLabel {
                weight: edge.label.weight,
                minlen: edge.label.minlen,
                ..EdgeLabel::new(edge.label.weight, edge.label.minlen)
            }
        };

        graph.set_edge(&edge.v, &edge.w, Some(simple_label), None);
    }

    graph
}

fn feasible_tree_ns(g: &mut DagreGraph) -> DagreGraph {
    let mut t = DagreGraph::new(false, false);

    if g.nodes.is_empty() {
        return t;
    }

    let start = g.nodes.keys().next().unwrap().clone();
    let size = g.nodes.len();
    t.set_node(&start, Some(NodeLabel::new(0.0, 0.0)));

    let mut iterations = 0;
    let max_iterations = size * 2;

    while tight_tree(&mut t, g) < size && iterations < max_iterations {
        iterations += 1;

        let mut min_slack = i32::MAX;
        let mut min_edge: Option<(String, String)> = None;

        for edge in g.edges.values() {
            if t.has_node(&edge.v) != t.has_node(&edge.w) {
                let s = slack(g, &edge.v, &edge.w);
                if s < min_slack {
                    min_slack = s;
                    min_edge = Some((edge.v.as_ref().to_string(), edge.w.as_ref().to_string()));
                }
            }
        }

        if let Some((v, w)) = min_edge {
            let delta = if t.has_node(&v) {
                slack(g, &v, &w)
            } else {
                -slack(g, &v, &w)
            };

            for node_v in t
                .nodes
                .keys()
                .map(|s| s.as_ref().to_string())
                .collect::<Vec<_>>()
            {
                if let Some(node) = g.node_mut(&node_v) {
                    let mut label = node.label.write();
                    if let Some(rank) = label.rank {
                        label.rank = Some(rank + delta);
                    }
                }
            }
        } else {
            break;
        }
    }

    t
}

fn init_low_lim_values(tree: &mut DagreGraph, root: &str) {
    let mut next_lim = 1;
    let mut visited = HashSet::new();

    enum StackItem {
        Enter(String, Option<String>),
        Exit(String, i32),
    }

    let mut stack = vec![StackItem::Enter(root.to_string(), None)];

    while let Some(item) = stack.pop() {
        match item {
            StackItem::Enter(v, parent) => {
                if !visited.contains(&v) {
                    visited.insert(v.clone());
                    let low = next_lim;

                    stack.push(StackItem::Exit(v.clone(), low));

                    for w in tree.neighbors(&v) {
                        if !visited.contains(&w) {
                            stack.push(StackItem::Enter(w, Some(v.clone())));
                        }
                    }

                    if let Some(node) = tree.node_mut(&v) {
                        node.label.write().low = Some(low);
                        if let Some(p) = parent {
                            node.label.write().parent = Some(crate::graph::arc_str(&p));
                        }
                    }
                }
            }
            StackItem::Exit(v, _low) => {
                if let Some(node) = tree.node_mut(&v) {
                    node.label.write().lim = Some(next_lim);
                    next_lim += 1;
                }
            }
        }
    }
}

fn init_cut_values(t: &mut DagreGraph, g: &DagreGraph) {
    let mut vs = Vec::new();
    let mut visited = HashSet::new();

    enum StackItem {
        NodeArray(Vec<String>),
        Node(String),
    }

    // Reverse nodes like TypeScript: Array.from(t.nodes.keys()).reverse()
    let mut start_nodes: Vec<String> = t.nodes.keys().map(|s| s.as_ref().to_string()).collect();
    start_nodes.reverse();
    let mut stack: Vec<StackItem> = vec![StackItem::NodeArray(start_nodes)];

    while let Some(item) = stack.pop() {
        match item {
            StackItem::NodeArray(mut nodes) => {
                if let Some(v) = nodes.pop() {
                    if !nodes.is_empty() {
                        stack.push(StackItem::NodeArray(nodes));
                    }

                    if !visited.contains(&v) {
                        visited.insert(v.clone());
                        let mut children: Vec<String> = t
                            .neighbors(&v)
                            .into_iter()
                            .filter(|w| !visited.contains(w))
                            .collect();

                        if !children.is_empty() {
                            stack.push(StackItem::Node(v));
                            children.reverse(); // Like TypeScript: children.reverse()
                            stack.push(StackItem::NodeArray(children));
                        } else {
                            vs.push(v);
                        }
                    }
                }
            }
            StackItem::Node(v) => {
                vs.push(v);
            }
        }
    }

    for i in 0..vs.len().saturating_sub(1) {
        let v = &vs[i];
        calculate_cut_value(t, g, v);
    }
}

fn calculate_cut_value(t: &mut DagreGraph, g: &DagreGraph, v: &str) {
    let parent = if let Some(node) = t.node(v) {
        node.label.read().parent.clone()
    } else {
        return;
    };

    if parent.is_none() {
        return;
    }

    let parent = parent.unwrap();

    let edge_exists = g.edge(v, &parent).is_some();
    let child_is_tail = edge_exists;

    let graph_edge_label = if edge_exists {
        g.edge(v, &parent).map(|e| e.label.clone())
    } else {
        g.edge(&parent, v).map(|e| e.label.clone())
    };

    let mut cutvalue = graph_edge_label.map(|l| l.weight).unwrap_or(0.0);

    if let Some(node) = g.node(v) {
        for edge_key in node.in_edges.iter().chain(node.out_edges.iter()) {
            if let Some(edge) = g.edge_by_key(edge_key) {
                let is_out_edge = edge.v.as_ref() == v;
                let other = if is_out_edge { &edge.w } else { &edge.v };

                if other.as_ref() != parent.as_ref() {
                    let points_to_head = is_out_edge == child_is_tail;
                    cutvalue += if points_to_head {
                        edge.label.weight
                    } else {
                        -edge.label.weight
                    };

                    if let Some(tree_edge) = t.edge(v, other) {
                        if let Some(other_cutvalue) = tree_edge.label.cutvalue {
                            cutvalue += if points_to_head {
                                -other_cutvalue
                            } else {
                                other_cutvalue
                            };
                        }
                    }
                }
            }
        }
    }

    if let Some(edge) = t.edge_mut(v, &parent) {
        edge.label.cutvalue = Some(cutvalue);
    } else if let Some(edge) = t.edge_mut(&parent, v) {
        edge.label.cutvalue = Some(cutvalue);
    }
}

fn leave_edge(tree: &DagreGraph) -> Option<(String, String)> {
    for edge in tree.edges.values() {
        if let Some(cutvalue) = edge.label.cutvalue {
            if cutvalue < 0.0 {
                return Some((edge.v.as_ref().to_string(), edge.w.as_ref().to_string()));
            }
        }
    }
    None
}

fn enter_edge(
    t: &DagreGraph,
    g: &DagreGraph,
    leave_v: &str,
    leave_w: &str,
) -> Option<(String, String)> {
    let (mut v, mut w) = (leave_v.to_string(), leave_w.to_string());

    if g.edge(&v, &w).is_none() {
        std::mem::swap(&mut v, &mut w);
    }

    let v_label = t.node(&v).unwrap().label.clone();
    let w_label = t.node(&w).unwrap().label.clone();

    let (tail_label, flip) = if v_label.read().lim.unwrap() > w_label.read().lim.unwrap() {
        (w_label, true)
    } else {
        (v_label, false)
    };

    let mut min_slack = i32::MAX;
    let mut min_edge: Option<(String, String)> = None;

    for edge in g.edges.values() {
        let v_node = t.node(&edge.v).unwrap();
        let w_node = t.node(&edge.w).unwrap();

        let v_desc = is_descendant(&v_node.label.read(), &tail_label.read());
        let w_desc = is_descendant(&w_node.label.read(), &tail_label.read());

        if flip == v_desc && flip != w_desc {
            let s = slack(g, &edge.v, &edge.w);
            if s < min_slack {
                min_slack = s;
                min_edge = Some((edge.v.as_ref().to_string(), edge.w.as_ref().to_string()));
            }
        }
    }

    min_edge
}

fn is_descendant(v_label: &NodeLabel, root_label: &NodeLabel) -> bool {
    root_label.low.unwrap() <= v_label.lim.unwrap()
        && v_label.lim.unwrap() <= root_label.lim.unwrap()
}

fn exchange_edges(
    t: &mut DagreGraph,
    g: &mut DagreGraph,
    leave_v: &str,
    leave_w: &str,
    enter_v: &str,
    enter_w: &str,
) {
    t.remove_edge(leave_v, leave_w, None);
    t.set_edge(enter_v, enter_w, Some(EdgeLabel::new(1.0, 1)), None);

    if let Some(root) = t.nodes.keys().next().cloned() {
        init_low_lim_values(t, &root);
        init_cut_values(t, g);

        // Use Vec to maintain insertion order (like TypeScript's Set -> Array.from)
        let mut visited_set = HashSet::new();
        let mut visited_vec = Vec::new();
        let mut stack = vec![root.clone()];

        while let Some(v) = stack.pop() {
            if !visited_set.contains(&v) {
                visited_set.insert(v.clone());
                visited_vec.push(v.clone());
                // Add neighbors in reverse order (like TypeScript)
                let neighbors: Vec<String> = t.neighbors(&v).into_iter().collect();
                for w in neighbors.iter().rev() {
                    if !visited_set.contains(w.as_str()) {
                        stack.push(crate::graph::arc_str(w));
                    }
                }
            }
        }

        let vs = visited_vec;
        for v in vs.iter().skip(1) {
            if let Some(node) = t.node(v) {
                if let Some(parent) = &node.label.read().parent {
                    let edge_exists = g.edge(v, parent).is_some();
                    let flipped = !edge_exists;

                    let edge_label = if edge_exists {
                        g.edge(v, parent).map(|e| e.label.clone())
                    } else {
                        g.edge(parent, v).map(|e| e.label.clone())
                    };

                    if let Some(edge_label) = edge_label {
                        let parent_rank = g.node(parent).and_then(|n| n.label.read().rank);
                        if let Some(parent_rank) = parent_rank {
                            let new_rank = if flipped {
                                parent_rank + edge_label.minlen
                            } else {
                                parent_rank - edge_label.minlen
                            };

                            if let Some(v_node) = g.node_mut(v) {
                                v_node.label.write().rank = Some(new_rank);
                            }
                        }
                    }
                }
            }
        }
    }
}
