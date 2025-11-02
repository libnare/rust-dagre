use crate::graph::DagreGraph;
use crate::types::EdgeKey;
use crate::utils::unique_id;
use ahash::AHashSet as HashSet;

enum StackItem {
    Node(String),
    Exit(String),
}

pub fn acyclic_run(g: &mut DagreGraph) {
    let mut edges_to_reverse = Vec::new();
    let mut visited = HashSet::new();
    let mut path = HashSet::new();
    let mut stack: Vec<StackItem> = g
        .nodes
        .keys()
        .rev()
        .map(|k| StackItem::Node(k.as_ref().to_string()))
        .collect();

    while let Some(item) = stack.pop() {
        match item {
            StackItem::Exit(v) => {
                path.remove(&v);
            }
            StackItem::Node(v) => {
                if !visited.contains(&v) {
                    visited.insert(v.clone());
                    path.insert(v.clone());
                    stack.push(StackItem::Exit(v.clone()));

                    if let Some(node) = g.node(&v) {
                        let out_edges: Vec<EdgeKey> = node.out_edges.to_vec();
                        for edge_key in out_edges.iter().rev() {
                            if let Some(edge) = g.edge_by_key(edge_key) {
                                if path.contains(edge.w.as_ref()) {
                                    edges_to_reverse.push(edge_key.clone());
                                }
                                stack.push(StackItem::Node(edge.w.as_ref().to_string()));
                            }
                        }
                    }
                }
            }
        }
    }

    for edge_key in edges_to_reverse {
        if let Some(edge) = g.edge_by_key(&edge_key).cloned() {
            let mut label = edge.label.clone();
            g.remove_edge_by_key(&edge_key);
            label.forward_name = edge.name.as_ref().map(|s| s.as_ref().to_string());
            label.reversed = Some(true);
            g.set_edge(&edge.w, &edge.v, Some(label), Some(&unique_id("rev")));
        }
    }
}

pub fn acyclic_undo(g: &mut DagreGraph) {
    let mut edges_to_restore = Vec::new();

    for edge in g.edges.values() {
        if edge.label.reversed == Some(true) {
            edges_to_restore.push((
                edge.key.clone(),
                edge.v.as_ref().to_string(),
                edge.w.as_ref().to_string(),
                edge.label.clone(),
            ));
        }
    }

    for (key, v, w, mut label) in edges_to_restore {
        if let Some(points) = &mut label.points {
            points.reverse();
        }
        g.remove_edge_by_key(&key);
        let forward_name = label.forward_name.clone();
        label.reversed = None;
        label.forward_name = None;
        g.set_edge(&w, &v, Some(label), forward_name.as_deref());
    }
}
