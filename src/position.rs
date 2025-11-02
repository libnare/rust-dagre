use crate::graph::DagreGraph;
use crate::types::{Alignment, EdgeLabel, LayoutConfig, NodeLabel};
use crate::utils::build_layer_matrix;
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use rayon::prelude::*;
use std::sync::Arc;

// Pre-extract node properties to avoid repeated lock acquisitions
struct NodeCache {
    width: f64,
    height: f64,
    dummy: Option<String>,
}

pub fn position(g: &mut DagreGraph, layout: &LayoutConfig) {
    let mut step_timer = std::time::Instant::now();

    // OPTIMIZATION: Build comprehensive node cache upfront
    let node_cache: HashMap<Arc<str>, NodeCache> = g
        .nodes
        .iter()
        .map(|(id, node)| {
            let label = node.label.read();
            let cache = NodeCache {
                width: label.width,
                height: label.height,
                dummy: label.dummy.clone(),
            };
            drop(label);
            (id.clone(), cache)
        })
        .collect();
    eprintln!("[POSITION] build node cache: {:?}", step_timer.elapsed());

    // Helper: Add conflict between two nodes
    let add_conflict =
        |conflicts: &mut HashMap<Arc<str>, HashSet<Arc<str>>>, v: Arc<str>, w: Arc<str>| {
            let (v_key, w_val) = if v.as_ref() > w.as_ref() {
                (w, v)
            } else {
                (v, w)
            };
            conflicts
                .entry(v_key)
                .or_insert_with(HashSet::new)
                .insert(w_val);
        };

    // Helper: Check if conflict exists (using &str to avoid clone)
    let has_conflict =
        |conflicts: &HashMap<Arc<str>, HashSet<Arc<str>>>, v: &Arc<str>, w: &Arc<str>| -> bool {
            let (v_ref, w_ref) = if v.as_ref() > w.as_ref() {
                (w, v)
            } else {
                (v, w)
            };
            conflicts
                .get(v_ref)
                .map_or(false, |set| set.contains(w_ref))
        };

    // Build block graph for horizontal compaction
    let build_block_graph = |g: &DagreGraph,
                             layout: &LayoutConfig,
                             layering: &[Vec<Arc<str>>],
                             root: &HashMap<Arc<str>, Arc<str>>,
                             reverse_sep: bool,
                             node_cache: &HashMap<Arc<str>, NodeCache>|
     -> DagreGraph {
        let node_sep = layout.nodesep.unwrap_or(50.0);
        let edge_sep = layout.edgesep.unwrap_or(10.0);
        let mut block_graph = DagreGraph::new(true, false);

        for layer in layering {
            let mut u: Option<&str> = None;
            for v in layer {
                if let Some(v_root) = root.get(v) {
                    block_graph.set_node(v_root, Some(NodeLabel::default()));

                    if let Some(u_val) = u {
                        if let (Some(u_root), Some(v_cache), Some(u_cache)) =
                            (root.get(u_val), node_cache.get(v), node_cache.get(u_val))
                        {
                            // Read from pre-built cache (lock-free)
                            let v_width = v_cache.width;
                            let w_width = u_cache.width;
                            let v_is_dummy = v_cache.dummy.is_some();
                            let w_is_dummy = u_cache.dummy.is_some();

                            // Only read labelpos from lock
                            let v_node = g.node(v).unwrap();
                            let u_node = g.node(u_val).unwrap();
                            let v_label = v_node.label.read();
                            let w_label = u_node.label.read();

                            let mut sum = 0.0;
                            let mut delta = 0.0;

                            sum += v_width / 2.0;
                            if let Some(ref labelpos) = v_label.labelpos {
                                delta = match labelpos.as_ref() {
                                    "l" => -v_width / 2.0,
                                    "r" => v_width / 2.0,
                                    _ => 0.0,
                                };
                            }
                            if delta != 0.0 {
                                sum += if reverse_sep { delta } else { -delta };
                            }

                            delta = 0.0;
                            sum += if v_is_dummy { edge_sep } else { node_sep } / 2.0;
                            sum += if w_is_dummy { edge_sep } else { node_sep } / 2.0;
                            sum += w_width / 2.0;

                            if let Some(ref labelpos) = w_label.labelpos {
                                delta = match labelpos.as_ref() {
                                    "l" => w_width / 2.0,
                                    "r" => -w_width / 2.0,
                                    _ => 0.0,
                                };
                            }
                            if delta != 0.0 {
                                sum += if reverse_sep { delta } else { -delta };
                            }

                            drop(v_label);
                            drop(w_label);

                            let current_weight = block_graph
                                .edge(u_root, v_root)
                                .map(|e| e.label.weight as f64)
                                .unwrap_or(0.0);
                            let max = sum.max(current_weight);

                            block_graph.set_edge(
                                u_root,
                                v_root,
                                Some(EdgeLabel {
                                    weight: max,
                                    minlen: 1,
                                    ..Default::default()
                                }),
                                None,
                            );
                        }
                    }
                    u = Some(v.as_ref());
                }
            }
        }

        block_graph
    };

    // Vertical alignment: align nodes into vertical blocks (OPTIMIZED with profiling)
    let vertical_alignment = |layering: &[Vec<Arc<str>>],
                              conflicts: &HashMap<Arc<str>, HashSet<Arc<str>>>,
                              neighbor_fn: &dyn Fn(&Arc<str>) -> Vec<Arc<str>>|
     -> (HashMap<Arc<str>, Arc<str>>, HashMap<Arc<str>, Arc<str>>) {
        let va_start = std::time::Instant::now();
        let total_nodes: usize = layering.iter().map(|layer| layer.len()).sum();
        let mut root = HashMap::with_capacity(total_nodes);
        let mut align = HashMap::with_capacity(total_nodes);
        let mut pos = HashMap::with_capacity(total_nodes);

        // Initialize root, align, and pos
        let init_start = std::time::Instant::now();
        for layer in layering {
            for (order, v) in layer.iter().enumerate() {
                // Minimize clones: reuse v reference
                root.insert(v.clone(), v.clone());
                align.insert(v.clone(), v.clone());
                pos.insert(v.clone(), order);
            }
        }
        let init_time = init_start.elapsed();

        // Process each layer
        let mut neighbor_time = std::time::Duration::ZERO;
        let mut sort_time = std::time::Duration::ZERO;
        let mut conflict_time = std::time::Duration::ZERO;
        let mut update_time = std::time::Duration::ZERO;

        for (_layer_idx, layer) in layering.iter().enumerate() {
            let mut prev_idx = -1i32;
            for v in layer {
                let n_start = std::time::Instant::now();
                let ws = neighbor_fn(v);
                neighbor_time += n_start.elapsed();

                if !ws.is_empty() {
                    let s_start = std::time::Instant::now();
                    // OPTIMIZATION: All nodes are in pos, no need for unwrap_or
                    let mut ws_array = ws;
                    ws_array.sort_unstable_by_key(|w| *pos.get(w).unwrap());
                    sort_time += s_start.elapsed();

                    let mp = (ws_array.len() - 1) as f64 / 2.0000001;
                    let il = mp.ceil() as usize;
                    for i in (mp.floor() as usize)..=il {
                        if i >= ws_array.len() {
                            break;
                        }
                        let w = &ws_array[i];
                        let w_pos = *pos.get(w).unwrap();

                        // Check if align.get(v) === v (like TypeScript)
                        let is_aligned = align.get(v).map(|s| s == v).unwrap_or(false);

                        let c_start = std::time::Instant::now();
                        let no_conflict = !has_conflict(conflicts, v, w);
                        conflict_time += c_start.elapsed();

                        if is_aligned && prev_idx < w_pos as i32 && no_conflict {
                            let u_start = std::time::Instant::now();
                            // OPTIMIZATION: Minimize String clones
                            if let Some(x) = root.get(w).cloned() {
                                align.insert(w.clone(), v.clone());
                                root.insert(v.clone(), x.clone());
                                align.insert(v.clone(), x);
                                prev_idx = w_pos as i32;
                            }
                            update_time += u_start.elapsed();
                        }
                    }
                }
            }
        }

        let total_va = va_start.elapsed();
        if layering.len() > 100 {
            eprintln!("[VA DETAIL] total={:?}, init={:?}, neighbor={:?}, sort={:?}, conflict={:?}, update={:?}", 
                total_va, init_time, neighbor_time, sort_time, conflict_time, update_time);
        }

        (root, align)
    };

    // Horizontal compaction: assign x-coordinates
    let horizontal_compaction = |g: &DagreGraph,
                                 layout: &LayoutConfig,
                                 layering: &[Vec<Arc<str>>],
                                 root: &HashMap<Arc<str>, Arc<str>>,
                                 align: &HashMap<Arc<str>, Arc<str>>,
                                 reverse_sep: bool,
                                 node_cache: &HashMap<Arc<str>, NodeCache>|
     -> HashMap<Arc<str>, f64> {
        let block_g = build_block_graph(g, layout, layering, root, reverse_sep, node_cache);

        let border_type = if reverse_sep {
            "borderLeft"
        } else {
            "borderRight"
        };
        let mut xs: HashMap<Arc<str>, f64> = HashMap::new();

        // First pass: place blocks with smallest possible coordinates
        if !block_g.nodes.is_empty() {
            let node_count = block_g.nodes.len();
            let mut stack: Vec<Arc<str>> = Vec::with_capacity(node_count * 2);
            stack.extend(block_g.nodes.keys().cloned());
            let mut visited = HashSet::with_capacity(node_count);

            while let Some(v) = stack.pop() {
                if visited.contains(&v) {
                    let mut max: f64 = 0.0;
                    if let Some(node) = block_g.node(&v) {
                        for edge_key in node.in_edges.iter() {
                            if let Some(edge) = block_g.edge_by_key(edge_key) {
                                let edge_weight = edge.label.weight;
                                let u_x = xs.get(&edge.v).copied().unwrap_or(0.0);
                                max = max.max(u_x + edge_weight);
                            }
                        }
                    }
                    xs.insert(v, max);
                } else {
                    visited.insert(v.clone());
                    stack.push(v.clone());
                    if let Some(predecessors) = block_g.predecessors(&v) {
                        for (pred, _) in predecessors {
                            stack.push(pred.clone());
                        }
                    }
                }
            }
        }

        // Second pass: remove unused space
        if !block_g.nodes.is_empty() {
            let node_count = block_g.nodes.len();
            let mut stack: Vec<Arc<str>> = Vec::with_capacity(node_count * 2);
            stack.extend(block_g.nodes.keys().cloned());
            let mut visited = HashSet::with_capacity(node_count);

            while let Some(v) = stack.pop() {
                if visited.contains(&v) {
                    let mut min = f64::INFINITY;
                    if let Some(node) = block_g.node(v.as_ref()) {
                        for edge_key in node.out_edges.iter() {
                            if let Some(edge) = block_g.edge_by_key(edge_key) {
                                let edge_weight = edge.label.weight;
                                let w_x = xs.get(&edge.w).copied().unwrap_or(0.0);
                                min = min.min(w_x - edge_weight);
                            }
                        }
                    }

                    // Read from cache
                    let is_not_dummy = node_cache.get(&v).map_or(false, |c| c.dummy.is_none());

                    if is_not_dummy && min != f64::INFINITY {
                        if let Some(node) = g.node(v.as_ref()) {
                            let label = node.label.read();
                            let has_border = label.border_type.as_deref() == Some(border_type);
                            drop(label);

                            if !has_border {
                                let current_x = xs.get(&v).copied().unwrap_or(0.0);
                                let new_x = current_x.max(min);
                                xs.insert(v.clone(), new_x);
                            }
                        }
                    }
                } else {
                    visited.insert(v.clone());
                    stack.push(v.clone());
                    if let Some(successors) = block_g.successors(v.as_ref()) {
                        for (succ, _) in successors {
                            stack.push(succ.clone());
                        }
                    }
                }
            }
        }

        // Assign x coordinates to all nodes
        for v in align.values() {
            if let Some(r) = root.get(v) {
                if let Some(&x) = xs.get(r) {
                    xs.insert(v.clone(), x);
                }
            }
        }

        xs
    };

    // Find Type 1 conflicts (PARALLELIZED)
    let find_type1_conflicts = |g: &DagreGraph,
                                layering: &[Vec<Arc<str>>],
                                node_cache: &HashMap<Arc<str>, NodeCache>|
     -> HashMap<Arc<str>, HashSet<Arc<str>>> {
        if layering.is_empty() {
            return HashMap::new();
        }

        // Process layer pairs in parallel
        let layer_pairs: Vec<usize> = (1..layering.len()).collect();
        let conflicts_vec: Vec<HashMap<Arc<str>, HashSet<Arc<str>>>> = layer_pairs
            .par_iter()
            .map(|&k| {
                let mut local_conflicts = HashMap::new();
                let prev = &layering[k - 1];
                let layer = &layering[k];
                let mut k0 = 0;
                let mut scan_pos = 0;
                let prev_layer_length = prev.len();
                let last_node: Option<&str> = layer.last().map(|s| s.as_ref());

                for i in 0..layer.len() {
                    let v = &layer[i];
                    let v_is_dummy = node_cache.get(v).map_or(false, |c| c.dummy.is_some());
                    let w = if v_is_dummy {
                        g.predecessors(v).and_then(|preds| {
                            preds.keys().find(|u| {
                                node_cache
                                    .get(u.as_ref())
                                    .map_or(false, |c| c.dummy.is_some())
                            })
                        })
                    } else {
                        None
                    };

                    if w.is_some() || Some(v.as_ref()) == last_node {
                        let k1 = w
                            .and_then(|w_val| g.node(w_val))
                            .map(|n| n.order_atomic.load(std::sync::atomic::Ordering::Relaxed))
                            .unwrap_or(prev_layer_length);

                        for j in scan_pos..=i {
                            let scan_node = &layer[j];

                            if let Some(predecessors) = g.predecessors(scan_node) {
                                for (u, _) in predecessors {
                                    if let Some(u_node) = g.node(&u) {
                                        let u_pos = Some(
                                            u_node
                                                .order_atomic
                                                .load(std::sync::atomic::Ordering::Relaxed),
                                        );
                                        let u_is_dummy = node_cache
                                            .get(u.as_ref())
                                            .map_or(false, |c| c.dummy.is_some());
                                        let scan_node_dummy = node_cache
                                            .get(scan_node)
                                            .map_or(false, |c| c.dummy.is_some());

                                        if let Some(u_pos_val) = u_pos {
                                            if (u_pos_val < k0 || k1 < u_pos_val)
                                                && !(u_is_dummy && scan_node_dummy)
                                            {
                                                add_conflict(
                                                    &mut local_conflicts,
                                                    u.clone(),
                                                    scan_node.clone(),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        scan_pos = i + 1;
                        k0 = k1;
                    }
                }
                local_conflicts
            })
            .collect();

        // Merge all conflicts
        let total_nodes: usize = layering.iter().map(|layer| layer.len()).sum();
        let mut conflicts = HashMap::with_capacity(total_nodes / 10);
        for local in conflicts_vec {
            for (key, value) in local {
                conflicts
                    .entry(key)
                    .or_insert_with(HashSet::new)
                    .extend(value);
            }
        }
        conflicts
    };

    // Find Type 2 conflicts (PARALLELIZED)
    let find_type2_conflicts = |g: &DagreGraph,
                                layering: &[Vec<Arc<str>>],
                                node_cache: &HashMap<Arc<str>, NodeCache>|
     -> HashMap<Arc<str>, HashSet<Arc<str>>> {
        if layering.is_empty() {
            return HashMap::new();
        }

        // Process layer pairs in parallel
        let layer_pairs: Vec<usize> = (1..layering.len()).collect();
        let conflicts_vec: Vec<HashMap<Arc<str>, HashSet<Arc<str>>>> = layer_pairs
            .par_iter()
            .map(|&i| {
                let mut local_conflicts = HashMap::new();

                let scan = |conflicts: &mut HashMap<Arc<str>, HashSet<Arc<str>>>,
                            south: &[Arc<str>],
                            south_pos: usize,
                            south_end: usize,
                            prev_north_border: i32,
                            next_north_border: i32| {
                    for idx in south_pos..south_end {
                        let v = &south[idx];
                        if node_cache.get(v).map_or(false, |c| c.dummy.is_some()) {
                            if let Some(predecessors) = g.predecessors(v) {
                                for (u, _) in predecessors {
                                    if let Some(u_node) = g.node(&u) {
                                        let order = u_node
                                            .order_atomic
                                            .load(std::sync::atomic::Ordering::Relaxed)
                                            as i32;
                                        let is_dummy = node_cache
                                            .get(u.as_ref())
                                            .map_or(false, |c| c.dummy.is_some());

                                        if order > 0 {
                                            if is_dummy && (order as i32) < prev_north_border
                                                || (order as i32) > next_north_border
                                            {
                                                add_conflict(conflicts, u.clone(), v.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                let north = &layering[i - 1];
                let south = &layering[i];
                let south_len = south.len();
                let mut prev_north_pos = -1i32;
                let mut next_north_pos = 0i32;
                let mut south_pos = 0;

                for south_lookahead in 0..south_len {
                    let v = &south[south_lookahead];
                    if node_cache
                        .get(v)
                        .map_or(false, |c| c.dummy.as_deref() == Some("border"))
                    {
                        if let Some(predecessors) = g.predecessors(v) {
                            if let Some((pred, _)) = predecessors.iter().next() {
                                if let Some(pred_node) = g.node(pred) {
                                    let order = pred_node
                                        .order_atomic
                                        .load(std::sync::atomic::Ordering::Relaxed);
                                    if order > 0 {
                                        next_north_pos = order as i32;
                                        scan(
                                            &mut local_conflicts,
                                            south,
                                            south_pos,
                                            south_lookahead,
                                            prev_north_pos,
                                            next_north_pos,
                                        );
                                        south_pos = south_lookahead;
                                        prev_north_pos = next_north_pos;
                                    }
                                }
                            }
                        }
                    }
                }
                scan(
                    &mut local_conflicts,
                    south,
                    south_pos,
                    south_len,
                    next_north_pos,
                    north.len() as i32,
                );
                local_conflicts
            })
            .collect();

        // Merge all conflicts
        let mut conflicts = HashMap::new();
        for local in conflicts_vec {
            for (key, value) in local {
                conflicts
                    .entry(key)
                    .or_insert_with(HashSet::new)
                    .extend(value);
            }
        }
        conflicts
    };

    // Main position algorithm
    // OPTIMIZATION: Skip as_non_compound_graph clone - POSITION doesn't use parent/children
    step_timer = std::time::Instant::now();
    let layering = build_layer_matrix(g);
    eprintln!("[POSITION] build_layer_matrix: {:?}", step_timer.elapsed());

    let ranksep = layout.ranksep.unwrap_or(50.0);

    // Assign Y coordinates based on rank
    step_timer = std::time::Instant::now();
    let mut y = 0.0;
    for layer in &layering {
        // Read height from cache (lock-free)
        let max_height = layer
            .iter()
            .filter_map(|v| node_cache.get(v))
            .map(|c| c.height)
            .fold(0.0f64, |a, b| a.max(b));

        for v in layer {
            if let Some(node) = g.node_mut(v) {
                let y_coord = y + max_height / 2.0;
                node.label.write().y = Some(y_coord);
            }
        }
        y += max_height + ranksep;
    }
    eprintln!("[POSITION] assign Y: {:?}", step_timer.elapsed());

    // Find conflicts
    step_timer = std::time::Instant::now();
    let type1_conflicts = find_type1_conflicts(g, &layering, &node_cache);
    eprintln!(
        "[POSITION] find_type1_conflicts: {:?}",
        step_timer.elapsed()
    );

    step_timer = std::time::Instant::now();
    let type2_conflicts = find_type2_conflicts(g, &layering, &node_cache);
    eprintln!(
        "[POSITION] find_type2_conflicts: {:?}",
        step_timer.elapsed()
    );

    step_timer = std::time::Instant::now();
    let mut conflicts = type1_conflicts;
    for (key, value) in type2_conflicts {
        conflicts
            .entry(key)
            .or_insert_with(HashSet::new)
            .extend(value);
    }
    eprintln!("[POSITION] merge conflicts: {:?}", step_timer.elapsed());

    // Calculate x coordinates for all 4 alignments in parallel
    step_timer = std::time::Instant::now();
    let alignments = vec![("u", "l"), ("u", "r"), ("d", "l"), ("d", "r")];

    let xss_vec: Vec<(String, HashMap<Arc<str>, f64>)> = alignments
        .par_iter()
        .map(|&(vertical, horizontal)| {
            let align_start = std::time::Instant::now();

            let clone_start = std::time::Instant::now();
            let mut adjusted_layering: Vec<Vec<Arc<str>>> = if vertical == "u" {
                layering.clone()
            } else {
                layering.iter().rev().cloned().collect()
            };

            if horizontal == "r" {
                adjusted_layering = adjusted_layering
                    .iter()
                    .map(|layer| layer.iter().rev().cloned().collect())
                    .collect();
            }
            let clone_time = clone_start.elapsed();

            let neighbor_start = std::time::Instant::now();
            // OPTIMIZATION: Pre-build neighbor cache to avoid repeated lookups and clones
            let neighbor_cache: HashMap<Arc<str>, Vec<Arc<str>>> = adjusted_layering
                .iter()
                .flatten()
                .map(|v| {
                    let neighbors = if vertical == "u" {
                        g.predecessors(v.as_ref())
                            .map(|preds| preds.keys().cloned().collect())
                            .unwrap_or_else(Vec::new)
                    } else {
                        g.successors(v.as_ref())
                            .map(|succs| succs.keys().cloned().collect())
                            .unwrap_or_else(Vec::new)
                    };
                    (v.clone(), neighbors)
                })
                .collect();

            let neighbor_fn: Box<dyn Fn(&Arc<str>) -> Vec<Arc<str>> + Send + Sync> = Box::new(move |v: &Arc<str>| {
                neighbor_cache.get(v).cloned().unwrap_or_else(Vec::new)
            });
            let neighbor_time = neighbor_start.elapsed();

            let valign_start = std::time::Instant::now();
            let (root, align) =
                vertical_alignment(&adjusted_layering, &conflicts, neighbor_fn.as_ref());
            let valign_time = valign_start.elapsed();

            let hcomp_start = std::time::Instant::now();
            let mut xs = horizontal_compaction(
                g,
                layout,
                &adjusted_layering,
                &root,
                &align,
                horizontal == "r",
                &node_cache,
            );
            let hcomp_time = hcomp_start.elapsed();

            if horizontal == "r" {
                xs = xs.into_iter().map(|(k, v)| (k, -v)).collect();
            }

            let total_time = align_start.elapsed();
            if vertical == "u" && horizontal == "l" {
                eprintln!("[POSITION DETAIL] alignment ul: total={:?}, clone={:?}, neighbor={:?}, valign={:?}, hcomp={:?}", 
                    total_time, clone_time, neighbor_time, valign_time, hcomp_time);
            }

            let alignment_key = format!("{}{}", vertical, horizontal);
            (alignment_key, xs)
        })
        .collect();

    let mut xss: HashMap<String, HashMap<Arc<str>, f64>> = xss_vec.into_iter().collect();
    eprintln!(
        "[POSITION] calculate alignments (parallel): {:?}",
        step_timer.elapsed()
    );

    // Find smallest width alignment (iterate in specific order to match TypeScript)
    step_timer = std::time::Instant::now();

    let mut min_width = f64::INFINITY;
    let mut min_key = String::new();
    for key in &["ul", "ur", "dl", "dr"] {
        if let Some(xs) = xss.get(*key) {
            let mut max = f64::NEG_INFINITY;
            let mut min = f64::INFINITY;
            for (v, &x) in xs {
                // Read width from cache (lock-free)
                if let Some(cache) = node_cache.get(v) {
                    let half_width = cache.width / 2.0;
                    max = max.max(x + half_width);
                    min = min.min(x - half_width);
                }
            }
            let width = max - min;
            if width < min_width {
                min_width = width;
                min_key = key.to_string();
            }
        }
    }
    eprintln!("[POSITION] find min width: {:?}", step_timer.elapsed());

    // Align all coordinates
    step_timer = std::time::Instant::now();
    let align_to = xss.get(&min_key).unwrap();
    let align_to_vals: Vec<f64> = align_to.values().copied().collect();
    let align_to_min = align_to_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let align_to_max = align_to_vals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    for vertical in &["u", "d"] {
        for horizontal in &["l", "r"] {
            let alignment = format!("{}{}", vertical, horizontal);
            if alignment != min_key {
                if let Some(xs) = xss.get_mut(&alignment) {
                    let vals: Vec<f64> = xs.values().copied().collect();
                    let vs_min = vals.iter().copied().fold(f64::INFINITY, f64::min);
                    let vs_max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                    let delta = if *horizontal == "l" {
                        align_to_min - vs_min
                    } else {
                        align_to_max - vs_max
                    };

                    if delta != 0.0 {
                        for v in xs.values_mut() {
                            *v += delta;
                        }
                    }
                }
            }
        }
    }
    eprintln!("[POSITION] align coordinates: {:?}", step_timer.elapsed());

    // Balance and assign final x coordinates
    step_timer = std::time::Instant::now();
    if let Some(align) = layout.align {
        let align_str = match align {
            Alignment::UL => "ul",
            Alignment::UR => "ur",
            Alignment::DL => "dl",
            Alignment::DR => "dr",
        };
        if let Some(xs) = xss.get(align_str) {
            for (v, &x) in xs {
                if let Some(node) = g.node_mut(v) {
                    node.label.write().x = Some(x);
                }
            }
        }
    } else {
        // Average of median two alignments
        if let Some(ul) = xss.get("ul") {
            for v in ul.keys() {
                let mut vals: Vec<f64> = vec![
                    *xss.get("ul").and_then(|m| m.get(v)).unwrap_or(&0.0),
                    *xss.get("ur").and_then(|m| m.get(v)).unwrap_or(&0.0),
                    *xss.get("dl").and_then(|m| m.get(v)).unwrap_or(&0.0),
                    *xss.get("dr").and_then(|m| m.get(v)).unwrap_or(&0.0),
                ];
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let x = (vals[1] + vals[2]) / 2.0;

                if let Some(node) = g.node_mut(v) {
                    node.label.write().x = Some(x);
                }
            }
        }
    }
    eprintln!("[POSITION] assign final X: {:?}", step_timer.elapsed());
}
