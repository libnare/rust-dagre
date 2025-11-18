use crate::graph::DagreGraph;
use crate::types::{EdgeKey, EdgeLabel, LayoutState, NodeLabel, Point};
use crate::utils::add_dummy_node;
use rayon::prelude::*;
use std::sync::Arc;

pub fn inject_edge_label_proxies(g: &mut DagreGraph) {
    let edges_to_process: Vec<(String, String, i32, i32, EdgeKey)> = g
        .edges
        .values()
        .filter_map(|e| {
            let edge = &e.label;
            if edge.width.is_some()
                && edge.height.is_some()
                && edge.width.unwrap() > 0.0
                && edge.height.unwrap() > 0.0
            {
                let v_node = g.node(&e.v)?;
                let w_node = g.node(&e.w)?;
                let v_rank = v_node.label.read().rank?;
                let w_rank = w_node.label.read().rank?;
                Some((
                    e.v.as_ref().to_string(),
                    e.w.as_ref().to_string(),
                    v_rank,
                    w_rank,
                    e.key.clone(),
                ))
            } else {
                None
            }
        })
        .collect();

    for (_v, _w, v_rank, w_rank, edge_key) in edges_to_process {
        let rank = (w_rank - v_rank) / 2 + v_rank;
        let mut label = NodeLabel::new(0.0, 0.0);
        label.rank = Some(rank);
        label.e = Some(edge_key);
        add_dummy_node(g, "edge-proxy", label, "_ep");
    }
}

pub fn remove_empty_ranks(g: &mut DagreGraph, state: &LayoutState) {
    if g.nodes.is_empty() {
        return;
    }

    let mut min_rank = i32::MAX;
    let mut max_rank = i32::MIN;

    for node in g.nodes.values() {
        if let Some(rank) = node.label.read().rank {
            min_rank = min_rank.min(rank);
            max_rank = max_rank.max(rank);
        }
    }

    let size = max_rank - min_rank + 1;
    if size > 0 {
        let mut layers: Vec<Option<Vec<String>>> = vec![None; size as usize];

        for node in g.nodes.values() {
            if let Some(rank) = node.label.read().rank {
                let idx = (rank - min_rank) as usize;
                layers[idx]
                    .get_or_insert_with(Vec::new)
                    .push(node.v.as_ref().to_string());
            }
        }

        let mut delta = 0;
        let node_rank_factor = state.node_rank_factor.unwrap_or(1.0) as i32;

        for (i, vs) in layers.iter().enumerate() {
            if vs.is_none() && (i as i32 % node_rank_factor) != 0 {
                delta -= 1;
            } else if delta != 0 && vs.is_some() {
                for v in vs.as_ref().unwrap() {
                    if let Some(node) = g.node_mut(v) {
                        if let Some(rank) = node.label.read().rank {
                            node.label.write().rank = Some(rank + delta);
                        }
                    }
                }
            }
        }
    }
}

pub fn normalize(g: &mut DagreGraph, state: &mut LayoutState) {
    let edge_count = g.edges.len();

    // Phase 1: Collect edges and calculate dummy nodes (parallel)
    type DummyData = (String, NodeLabel, bool); // (id, label, is_chain_start)
    type EdgeData = (String, String, EdgeLabel, Option<String>); // (v, w, label, name)
    type NormalizeData = (
        Vec<DummyData>, // dummy nodes to add
        Vec<EdgeData>,  // edges to add
        String,         // original edge v
        String,         // original edge w
        Option<String>, // original edge name
    );

    let edges_to_process: Vec<(String, String, Option<String>, EdgeLabel, i32, i32)> = g
        .edges
        .values()
        .filter_map(|e| {
            let v_node = g.node(&e.v)?;
            let w_node = g.node(&e.w)?;
            let v_label = v_node.label.read();
            let w_label = w_node.label.read();
            let v_rank = v_label.rank?;
            let w_rank = w_label.rank?;
            drop(v_label);
            drop(w_label);

            if w_rank != v_rank + 1 {
                Some((
                    e.v.as_ref().to_string(),
                    e.w.as_ref().to_string(),
                    e.name.as_ref().map(|s| s.as_ref().to_string()),
                    e.label.clone(),
                    v_rank,
                    w_rank,
                ))
            } else {
                None
            }
        })
        .collect();

    // Phase 2: Calculate all dummy nodes and edges in parallel
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dummy_counter = AtomicUsize::new(0);

    let edge_label_string = "edge-label".to_string();
    let normalize_data: Vec<NormalizeData> = edges_to_process
        .par_iter()
        .map(|(orig_v, w, name, edge_label, v_rank, w_rank)| {
            let mut dummies = Vec::new();
            let mut edges = Vec::new();
            let mut current_v = orig_v.clone();
            let mut rank = v_rank + 1;
            let edge_obj = EdgeKey::new(
                crate::graph::arc_str(&orig_v),
                crate::graph::arc_str(&w),
                name.as_ref().map(|s| crate::graph::arc_str(s)),
            );
            let mut is_first = true;

            while rank < *w_rank {
                let dummy_id = format!("_d{}", dummy_counter.fetch_add(1, Ordering::Relaxed));

                let mut attrs = NodeLabel::new(0.0, 0.0);
                attrs.rank = Some(rank);
                attrs.edge_label = Some(Box::new(edge_label.clone()));
                attrs.edge_obj = Some(edge_obj.clone());
                attrs.dummy = Some("edge".to_string());

                // Update for edge-label
                if Some(rank) == edge_label.label_rank {
                    attrs.width = edge_label.width.unwrap_or(0.0);
                    attrs.height = edge_label.height.unwrap_or(0.0);
                    attrs.dummy = Some(edge_label_string.clone());
                    attrs.labelpos = edge_label.labelpos.clone();
                }

                dummies.push((dummy_id.clone(), attrs, is_first));
                edges.push((
                    current_v.clone(),
                    dummy_id.clone(),
                    EdgeLabel::new(edge_label.weight, 1),
                    name.clone(),
                ));

                current_v = dummy_id;
                rank += 1;
                is_first = false;
            }

            edges.push((
                current_v,
                w.clone(),
                EdgeLabel::new(edge_label.weight, 1),
                name.clone(),
            ));

            (dummies, edges, orig_v.clone(), w.clone(), name.clone())
        })
        .collect();

    // Phase 3: Apply changes to graph (sequential but optimized)
    state.dummy_chains = Some(Vec::with_capacity(edge_count / 4));

    // First pass: remove original edges
    for (_, _, orig_v, w, name) in &normalize_data {
        g.remove_edge(orig_v, w, name.as_deref());
    }

    // Second pass: add all dummy nodes and edges (OPTIMIZED - move instead of clone)
    let mut all_edges = Vec::new();
    for (dummies, edges, _, _, _) in normalize_data.into_iter() {
        for (dummy_id, label, is_chain_start) in dummies {
            g.set_node(&dummy_id, Some(label)); // Move label (no clone!)
            if is_chain_start {
                state
                    .dummy_chains
                    .as_mut()
                    .unwrap()
                    .push(crate::graph::arc_str(&dummy_id));
            }
        }
        all_edges.push(edges);
    }

    // Third pass: add all edges (OPTIMIZED: use set_edge_fast since nodes already exist)
    for edges in all_edges {
        for (v, w, label, name) in edges {
            g.set_edge_fast(&v, &w, label, name.as_deref()); // No node creation overhead!
        }
    }
}

pub fn denormalize(g: &mut DagreGraph, state: &LayoutState) {
    use std::sync::Mutex;

    // Pre-allocate with estimated capacity
    let all_removed_nodes: Mutex<ahash::AHashSet<Arc<str>>> = Mutex::new(ahash::AHashSet::new());
    let all_removed_edges: Mutex<ahash::AHashSet<EdgeKey>> = Mutex::new(ahash::AHashSet::new());

    if let Some(dummy_chains) = &state.dummy_chains {
        type ChainData = (
            String,
            String,
            Option<String>,
            EdgeLabel,
            Vec<Arc<str>>,
            Vec<EdgeKey>,
        );
        let chain_data: Vec<ChainData> = dummy_chains
            .par_iter()
            .filter_map(|chain_start| {
                let mut v = chain_start.as_ref().to_string();
                let mut nodes_to_remove = Vec::new();
                let mut edges_to_remove = Vec::new();

                let (edge_v, edge_w, edge_name, mut edge_label) = {
                    let node = g.node(&v)?;
                    let label = node.label.read();
                    let edge_obj = label.edge_obj.as_ref()?;
                    let edge_label = label.edge_label.as_ref()?.as_ref().clone();
                    Some((
                        edge_obj.v.as_ref().to_string(),
                        edge_obj.w.as_ref().to_string(),
                        edge_obj.name.as_ref().map(|s| s.as_ref().to_string()),
                        edge_label,
                    ))
                }?;

                let mut loop_count = 0;
                loop {
                    loop_count += 1;
                    if loop_count > 10000 {
                        break;
                    }

                    let node = g.node(&v)?;
                    let label = node.label.clone();
                    let label_guard = label.read();

                    if label_guard.dummy.is_none() {
                        break;
                    }

                    // Collect point if exists
                    if let (Some(x), Some(y)) = (label_guard.x, label_guard.y) {
                        edge_label
                            .points
                            .get_or_insert_with(Vec::new)
                            .push(Point { x, y });
                    }

                    // Check for edge-label
                    if label_guard.dummy.as_deref() == Some("edge-label") {
                        edge_label.x = label_guard.x;
                        edge_label.y = label_guard.y;
                        edge_label.width = Some(label_guard.width);
                        edge_label.height = Some(label_guard.height);
                    }
                    drop(label_guard);

                    // Collect edges to remove
                    for edge_key in node.in_edges.iter().chain(node.out_edges.iter()) {
                        edges_to_remove.push(edge_key.clone());
                    }
                    nodes_to_remove.push(crate::graph::arc_str(&v));

                    // Get next node
                    let w = g
                        .successors(&v)?
                        .iter()
                        .find(|(k, _)| !g.removed_nodes.contains(k.as_ref()))
                        .map(|(k, _)| k.clone())?;

                    v = w.as_ref().to_string();
                }

                Some((
                    edge_v,
                    edge_w,
                    edge_name,
                    edge_label,
                    nodes_to_remove,
                    edges_to_remove,
                ))
            })
            .collect();

        // Phase 2: Apply changes sequentially
        for (edge_v, edge_w, edge_name, edge_label, nodes_to_remove, edges_to_remove) in chain_data
        {
            g.set_edge(&edge_v, &edge_w, Some(edge_label), edge_name.as_deref());

            // Mark nodes and edges for removal (consume vectors to avoid clone)
            for node in nodes_to_remove {
                g.removed_nodes.insert(node.clone());
                all_removed_nodes.lock().unwrap().insert(node);
            }

            for edge in edges_to_remove {
                all_removed_edges.lock().unwrap().insert(edge);
            }
        }
    }

    // Clean up edges and node references
    let removed_nodes = all_removed_nodes.into_inner().unwrap();
    let removed_edges = all_removed_edges.into_inner().unwrap();

    g.edges.retain(|k, _| !removed_edges.contains(k));

    // OPTIMIZATION: Parallel cleaning of node references
    // Note: IndexMap doesn't support par_iter_mut, so we keep sequential
    for node in g.nodes.values_mut() {
        Arc::make_mut(&mut node.in_edges).retain(|k| !removed_edges.contains(k));
        Arc::make_mut(&mut node.out_edges).retain(|k| !removed_edges.contains(k));
        node.predecessors
            .retain(|k, _| !removed_nodes.contains(k.as_ref()));
        node.successors
            .retain(|k, _| !removed_nodes.contains(k.as_ref()));
    }

    // Physically remove nodes at the end
    // UNSAFE OPTIMIZATION: Parallel filtering via Vec conversion
    // SAFETY: This is the final cleanup, we can reconstruct the IndexMap
    use rayon::prelude::*;

    // Convert to Vec, filter in parallel, reconstruct IndexMap
    let nodes_vec: Vec<(String, crate::graph::GraphNode)> = g
        .nodes
        .drain(..)
        .map(|(k, v)| (k.as_ref().to_string(), v))
        .collect();
    let filtered: Vec<(String, crate::graph::GraphNode)> = nodes_vec
        .into_par_iter()
        .filter(|(k, _)| !g.removed_nodes.contains(&crate::graph::arc_str(k)))
        .collect();

    // Reconstruct IndexMap from filtered results
    g.nodes = filtered
        .into_iter()
        .map(|(k, v)| (crate::graph::arc_str(&k), v))
        .collect();
    g.removed_nodes.clear();
}

pub fn remove_edge_label_proxies(g: &mut DagreGraph) {
    let nodes_to_remove: Vec<(String, i32, EdgeKey)> = g
        .nodes
        .values()
        .filter_map(|node| {
            // Read lock once and extract all needed values
            let label = node.label.read();
            if label.dummy.as_deref() == Some("edge-proxy") {
                let rank = label.rank.unwrap();
                let edge_key = label.e.as_ref().unwrap().clone();
                drop(label);
                Some((node.v.as_ref().to_string(), rank, edge_key))
            } else {
                None
            }
        })
        .collect();

    for (v, rank, edge_key) in nodes_to_remove {
        if let Some(edge) = g.edge_by_key(&edge_key) {
            let edge_v = edge.v.as_ref().to_string();
            let edge_w = edge.w.as_ref().to_string();
            if let Some(edge) = g.edge_mut(&edge_v, &edge_w) {
                edge.label.label_rank = Some(rank);
            }
        }
        g.remove_node(&v);
    }
}
