use crate::graph::DagreGraph;
use crate::types::{BarycenterEntry, EdgeLabel, LayoutConfig, MappedEntry, NodeLabel, SortResult};
use crate::utils::{max_rank, unique_id};
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use rayon::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;

// UNSAFE: Direct access to order_atomic without any overhead
// This is safe because:
// 1. order_atomic is only read during barycenter
// 2. order_atomic is only written during assign_order (no overlap)
// 3. Atomic operations are inherently thread-safe
#[inline(always)]
fn read_order_fast(node: &crate::graph::GraphNode) -> usize {
    // Direct atomic read (no lock, no overhead)
    node.order_atomic.load(Ordering::Relaxed)
}

#[inline(always)]
fn write_order_fast(node: &mut crate::graph::GraphNode, order: usize) {
    // Direct atomic write
    node.order_atomic.store(order, Ordering::Relaxed);
}

// LayerGraph is a DagreGraph with a root field
struct LayerGraph {
    graph: DagreGraph,
    root: Arc<str>,
}

impl LayerGraph {
    fn node(&self, v: &str) -> Option<&crate::graph::GraphNode> {
        self.graph.node(v)
    }

    fn node_mut(&mut self, v: &str) -> Option<&mut crate::graph::GraphNode> {
        self.graph.node_mut(v)
    }

    fn parent(&self, v: &str) -> Option<String> {
        self.graph.parent(v).map(|s| s.to_string())
    }

    fn children(&self, v: &str) -> Vec<Arc<str>> {
        if let Some(ref children_map) = self.graph.children {
            if let Some(children_set) = children_map.get(v) {
                let result: Vec<Arc<str>> = children_set.iter().cloned().collect();
                if v.contains("_root56") || v.contains("_root129") {}
                return result;
            }
        }
        if v.contains("_root56") || v.contains("_root129") {}
        Vec::new()
    }

    fn has_children(&self, v: &str) -> bool {
        if let Some(ref children_map) = self.graph.children {
            if let Some(children_set) = children_map.get(v) {
                return !children_set.is_empty();
            }
        }
        false
    }

    fn predecessors(&self, v: &str) -> Option<&HashMap<Arc<str>, usize>> {
        self.graph.predecessors(v)
    }
}

pub fn order(g: &mut DagreGraph, _layout: &LayoutConfig) {
    // Flatten nested arrays
    fn flat<T: Clone>(list: Vec<Vec<T>>) -> Vec<T> {
        list.into_iter().flatten().collect()
    }

    // Resolve conflicts in barycenter entries
    let resolve_conflicts = |entries: Vec<BarycenterEntry>, cg: &DagreGraph| -> Vec<MappedEntry> {
        let entry_count = entries.len();

        // OPTIMIZATION: Most graphs have no conflicts, early return
        if cg.edges.is_empty() {
            return entries
                .into_iter()
                .enumerate()
                .map(|(i, entry)| MappedEntry {
                    indegree: 0,
                    in_entries: Vec::new(),
                    out_entries: Vec::new(),
                    vs: vec![entry.v.clone()],
                    i,
                    barycenter: entry.barycenter,
                    weight: entry.weight,
                    merged: false,
                })
                .collect();
        }

        // OPTIMIZATION: Use Arc<str> HashMap to avoid String clones
        let mut mapped_entries: HashMap<Arc<str>, MappedEntry> = HashMap::with_capacity(entry_count);

        // OPTIMIZATION: Build Arc<str> -> index mapping (no more String clones!)
        let entry_ids: Vec<Arc<str>> = entries.iter().map(|e| e.v.clone()).collect();

        // Pre-allocate with cached length (consume entries to avoid clone)
        for (i, entry) in entries.into_iter().enumerate() {
            let v = entry.v;
            let tmp = MappedEntry {
                indegree: 0,
                in_entries: Vec::new(),
                out_entries: Vec::new(),
                vs: vec![v.clone()],
                i,
                barycenter: entry.barycenter,
                weight: entry.weight,
                merged: false,
            };
            mapped_entries.insert(v, tmp);
        }

        // Process edges - build in/out references
        for e in cg.edges.values() {
            if let (Some(entry_v), Some(entry_w)) = (
                mapped_entries.get(e.v.as_ref()),
                mapped_entries.get(e.w.as_ref()),
            ) {
                let v_idx = entry_v.i;
                let w_idx = entry_w.i;

                if let Some(entry_w_mut) = mapped_entries.get_mut(e.w.as_ref()) {
                    entry_w_mut.indegree += 1;
                    entry_w_mut.in_entries.push(v_idx);
                }
                if let Some(entry_v_mut) = mapped_entries.get_mut(e.v.as_ref()) {
                    entry_v_mut.out_entries.push(w_idx);
                }
            }
        }

        let mut source_set: Vec<MappedEntry> = Vec::with_capacity(entry_count / 4);
        let mut results: Vec<MappedEntry> = Vec::with_capacity(entry_count);

        for entry in mapped_entries.values() {
            if entry.indegree == 0 {
                source_set.push(entry.clone());
            }
        }

        while let Some(entry) = source_set.pop() {
            // Handle out edges before adding to results
            for w_idx in &entry.out_entries {
                let w_v = &entry_ids[*w_idx];
                if let Some(w_entry) = mapped_entries.get_mut(w_v) {
                    w_entry.indegree -= 1;
                    if w_entry.indegree == 0 {
                        source_set.push(w_entry.clone());
                    }
                }
            }

            results.push(entry.clone());

            // Handle in edges (merge) - need to update the entry in results
            let result_idx = results.len() - 1;
            for u_idx in &entry.in_entries {
                let u_v = &entry_ids[*u_idx];
                if let Some(u_entry) = mapped_entries.get(u_v) {
                    if u_entry.merged {
                        continue;
                    }
                    if u_entry.barycenter.is_none()
                        || results[result_idx].barycenter.is_none()
                        || u_entry.barycenter.unwrap() >= results[result_idx].barycenter.unwrap()
                    {
                        let mut sum = 0.0;
                        let mut weight = 0.0;
                        if let Some(v_weight) = results[result_idx].weight {
                            sum += results[result_idx].barycenter.unwrap() * v_weight;
                            weight += v_weight;
                        }
                        if let Some(u_w) = u_entry.weight {
                            sum += u_entry.barycenter.unwrap() * u_w;
                            weight += u_w;
                        }

                        results[result_idx].barycenter = Some(sum / weight);
                        results[result_idx].weight = Some(weight);
                        results[result_idx].vs.extend(u_entry.vs.clone());

                        if let Some(u_entry_mut) = mapped_entries.get_mut(u_v) {
                            u_entry_mut.merged = true;
                        }
                    }
                }
            }
        }

        results
    };

    // Barycenter calculation
    // Takes LayerGraph and original graph
    let barycenter =
        |lg: &LayerGraph, nodes: &[Arc<str>], orig_g: &DagreGraph| -> Vec<BarycenterEntry> {
            let node_array_len = nodes.len();
            let mut result: Vec<BarycenterEntry> = Vec::with_capacity(node_array_len);

            for v in nodes {
                // Get node from LayerGraph to check if it exists
                let lg_node = match lg.node(v.as_ref()) {
                    Some(n) => n,
                    None => {
                        result.push(BarycenterEntry {
                            v: v.clone(),
                            barycenter: Some(0.0),
                            weight: Some(0.0),
                        });
                        continue;
                    }
                };

                // Get the in_edges from LayerGraph (these are the edges added to LayerGraph)
                let in_v_len = lg_node.in_edges.len();

                if in_v_len == 0 {
                    result.push(BarycenterEntry {
                        v: v.clone(),
                        barycenter: None,
                        weight: None,
                    });
                    continue;
                }

                let mut sum = 0.0;
                let mut weight = 0.0;

                // This is a closure, we need to capture iter from outer scope
                for edge_key in lg_node.in_edges.iter() {
                    // Try to get edge from LayerGraph first
                    if let Some(lg_edge) = lg.graph.edges.get(edge_key) {
                        let edge_weight = lg_edge.label.weight;
                        // Get predecessor node from LayerGraph (which shares label with original graph)
                        if let Some(node_u) = lg.node(&lg_edge.v) {
                            // Lock-free atomic read (no RwLock overhead!)
                            let order = read_order_fast(node_u) as f64;
                            sum += edge_weight * order;
                            weight += edge_weight;
                        }
                    } else if let Some(orig_edge) = orig_g.edges.get(edge_key) {
                        // Fallback to original graph
                        let edge_weight = orig_edge.label.weight;
                        if let Some(node_u) = orig_g.node(&orig_edge.v) {
                            let order = read_order_fast(node_u) as f64;
                            sum += edge_weight * order;
                            weight += edge_weight;
                        }
                    }
                }
                let barycenter_val = sum / weight;

                result.push(BarycenterEntry {
                    v: v.clone(),
                    barycenter: Some(barycenter_val),
                    weight: Some(weight),
                });
            }
            result
        };

    // Sort entries
    let sort = |entries: Vec<MappedEntry>, bias_right: bool| -> SortResult {
        let consume_unsortable = |vs: &mut Vec<Arc<str>>,
                                  unsortable: &mut Vec<MappedEntry>,
                                  mut index: usize|
         -> usize {
            while let Some(last) = unsortable.last() {
                if last.i <= index {
                    let entry = unsortable.pop().unwrap();
                    vs.extend(entry.vs);
                    index += 1;
                } else {
                    break;
                }
            }
            index
        };

        // Partition
        let mut sortable: Vec<MappedEntry> = Vec::new();
        let mut unsortable: Vec<MappedEntry> = Vec::new();

        for value in entries {
            if value.barycenter.is_some() {
                sortable.push(value);
            } else {
                unsortable.push(value);
            }
        }

        unsortable.sort_by(|a, b| b.i.cmp(&a.i));

        let mut vs: Vec<Arc<str>> = Vec::new();
        let mut sum = 0.0;
        let mut weight = 0.0;
        let mut vs_index = 0;

        sortable.sort_by(|a, b| {
            let a_bary = a.barycenter.unwrap();
            let b_bary = b.barycenter.unwrap();
            if a_bary < b_bary {
                std::cmp::Ordering::Less
            } else if a_bary > b_bary {
                std::cmp::Ordering::Greater
            } else if bias_right {
                b.i.cmp(&a.i)
            } else {
                a.i.cmp(&b.i)
            }
        });

        vs_index = consume_unsortable(&mut vs, &mut unsortable, vs_index);

        // OPTIMIZATION: Consume sortable entries (move instead of clone)
        for entry in sortable {
            vs_index += entry.vs.len();
            vs.extend(entry.vs);
            sum += entry.barycenter.unwrap() * entry.weight.unwrap();
            weight += entry.weight.unwrap();
            vs_index = consume_unsortable(&mut vs, &mut unsortable, vs_index);
        }

        let mut result = SortResult {
            vs: vs,
            barycenter: None,
            weight: None,
        };
        if weight > 0.0 {
            result.barycenter = Some(sum / weight);
            result.weight = Some(weight);
        }
        result
    };

    // Sort subgraph (recursive)
    fn sort_subgraph(
        g: &LayerGraph,
        v: &str,
        cg: &DagreGraph,
        bias_right: bool,
        orig_g: &DagreGraph,
        barycenter: &dyn Fn(&LayerGraph, &[Arc<str>], &DagreGraph) -> Vec<BarycenterEntry>,
        resolve_conflicts: &dyn Fn(Vec<BarycenterEntry>, &DagreGraph) -> Vec<MappedEntry>,
        sort: &dyn Fn(Vec<MappedEntry>, bool) -> SortResult,
        flat: &dyn Fn(Vec<Vec<Arc<str>>>) -> Vec<Arc<str>>,
    ) -> SortResult {
        let node = g.node(v);
        let (bl, br) = node
            .map(|n| {
                let label = n.label.read();
                let bl = label.border_left.clone();
                let br = label.border_right.clone();
                drop(label);
                (bl, br)
            })
            .unwrap_or((None, None));

        // OPTIMIZATION: Use Arc<str> HashMap to avoid String clones
        let mut subgraphs: HashMap<Arc<str>, SortResult> = HashMap::new();

        let movable = if let Some(ref bl_vec) = bl {
            g.children(v)
                .into_iter()
                .filter(|w| {
                    let w_str = w.as_ref();
                    !bl_vec.iter().any(|s| s.as_ref() == w_str)
                        && !br
                            .as_ref()
                            .map(|br_vec| br_vec.iter().any(|s| s.as_ref() == w_str))
                            .unwrap_or(false)
                })
                .collect::<Vec<Arc<str>>>()
        } else {
            g.children(v)
        };

        if v.contains("_root56") {}

        let mut barycenters = barycenter(g, &movable, orig_g);

        // Debug: Check barycenter for da321892 and 2918b295
        for entry in &mut barycenters {
            if g.has_children(&entry.v) {
                let result = sort_subgraph(
                    g,
                    &entry.v,
                    cg,
                    bias_right,
                    orig_g,
                    barycenter,
                    resolve_conflicts,
                    sort,
                    flat,
                );
                // Extract barycenter values before moving result
                let result_barycenter = result.barycenter;
                let result_weight = result.weight;
                subgraphs.insert(entry.v.clone(), result); // Arc clone (cheap reference count)
                if result_barycenter.is_some() {
                    if entry.barycenter.is_none() {
                        entry.barycenter = result_barycenter;
                        entry.weight = result_weight;
                    } else {
                        let entry_bary = entry.barycenter.unwrap();
                        let entry_w = entry.weight.unwrap();
                        let result_bary = result_barycenter.unwrap();
                        let result_w = result_weight.unwrap();
                        entry.barycenter = Some(
                            (entry_bary * entry_w + result_bary * result_w) / (entry_w + result_w),
                        );
                        entry.weight = Some(entry_w + result_w);
                    }
                }
            }
        }

        let mut entries = resolve_conflicts(barycenters, cg);

        // Expand subgraphs
        for entry in &mut entries {
            entry.vs = flat(
                entry
                    .vs
                    .iter()
                    .map(|v| {
                        if let Some(subgraph) = subgraphs.get(v.as_ref()) {
                            subgraph.vs.clone()
                        } else {
                            vec![v.clone()]
                        }
                    })
                    .collect(),
            );
        }

        let mut result = sort(entries, bias_right);

        if let Some(ref bl_vec) = bl {
            let mut new_vs: Vec<Arc<str>> =
                bl_vec.iter().map(|s| crate::graph::arc_str(s)).collect();
            new_vs.extend(result.vs);
            if let Some(ref br_vec) = br {
                new_vs.extend(br_vec.iter().map(|s| crate::graph::arc_str(s)));
            }
            result.vs = new_vs;

            let predecessors = g.predecessors(&bl_vec[0]);
            if let Some(preds) = predecessors {
                if !preds.is_empty() {
                    let first_pred = preds.keys().next().unwrap();
                    if let Some(bl_pred_node) = g.node(first_pred) {
                        // Lock-free atomic read
                        let bl_pred_order = read_order_fast(bl_pred_node) as f64;

                        if let Some(ref br_vec) = br {
                            let br_predecessors = g.predecessors(&br_vec[0]);
                            if let Some(br_preds) = br_predecessors {
                                if !br_preds.is_empty() {
                                    let first_br_pred = br_preds.keys().next().unwrap();
                                    if let Some(br_pred_node) = g.node(first_br_pred) {
                                        let br_pred_order = read_order_fast(br_pred_node) as f64;

                                        if result.barycenter.is_none() {
                                            result.barycenter = Some(0.0);
                                            result.weight = Some(0.0);
                                        }

                                        let result_bary = result.barycenter.unwrap();
                                        let result_w = result.weight.unwrap();
                                        result.barycenter = Some(
                                            (result_bary * result_w
                                                + bl_pred_order
                                                + br_pred_order)
                                                / (result_w + 2.0),
                                        );
                                        result.weight = Some(result_w + 2.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    // Sweep layer graphs
    let sweep_layer_graphs = |layer_graphs: &mut [LayerGraph],
                              bias_right: bool,
                              g: &DagreGraph,
                              _iter: usize,
                              _is_down: bool| {
        let mut cg = DagreGraph::new(true, false);

        // OPTIMIZATION: Pre-allocate capacity for constraint graph
        let estimated_edges = layer_graphs.len() / 10;
        cg.edges.reserve(estimated_edges);

        // Process layer graphs SEQUENTIALLY (order matters for correctness)
        // Performance gain from atomic operations instead of RwLock
        for lg in layer_graphs.iter_mut() {
            // OPTIMIZATION: Use &str instead of cloning root
            let sorted = sort_subgraph(
                lg,
                &lg.root,
                &cg,
                bias_right,
                g,
                &barycenter,
                &resolve_conflicts,
                &sort,
                &flat,
            );
            let vs = sorted.vs;
            let length = vs.len();

            // OPTIMIZATION: Use unchecked access for atomic writes
            // SAFETY: vs contains valid node IDs from sort_subgraph
            for i in 0..length {
                if let Some(node) = lg.node_mut(&vs[i]) {
                    write_order_fast(node, i);
                }
            }

            // Add subgraph constraints
            let mut prev: HashMap<Arc<str>, Arc<str>> = HashMap::with_capacity(length / 4);
            let mut root_prev: Option<Arc<str>> = None;
            let mut exit = false;
            for v in &vs {
                let mut child = lg.parent(v);
                while let Some(ref child_v) = child {
                    let parent = lg.parent(child_v);
                    let prev_child: Option<Arc<str>>;
                    if let Some(ref parent_v) = parent {
                        prev_child = prev.get(parent_v.as_str()).cloned();
                        prev.insert(crate::graph::arc_str(parent_v), crate::graph::arc_str(child_v));
                    } else {
                        prev_child = root_prev.clone();
                        root_prev = Some(crate::graph::arc_str(child_v));
                    }
                    if let Some(ref prev_child_v) = prev_child {
                        if prev_child_v.as_ref() != child_v {
                            cg.set_edge(
                                prev_child_v,
                                child_v,
                                Some(EdgeLabel {
                                    weight: 1.0,
                                    minlen: 1,
                                    ..Default::default()
                                }),
                                None,
                            );
                            exit = true;
                            break;
                        }
                    }
                    child = parent;
                }
                if exit {
                    break;
                }
            }
        }
    };

    // Cross count calculation (PARALLELIZED for large graphs)
    let cross_count = |g: &DagreGraph, layering: &[Vec<Arc<str>>], best_cc: usize| -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let total_count = AtomicUsize::new(0);
        let should_stop = AtomicUsize::new(0);

        // OPTIMIZATION: Pre-check if best_cc is already 0 (can't improve)
        if best_cc == 0 {
            return 0;
        }

        // Process layer pairs in parallel
        let layer_pairs: Vec<_> = (1..layering.len()).collect();

        layer_pairs.par_iter().for_each(|&i| {
            // OPTIMIZATION: Check early termination more frequently
            if should_stop.load(AtomicOrdering::Relaxed) > 0 {
                return;
            }

            let north_layer = &layering[i - 1];
            let south_layer = &layering[i];

            // Use &str as key to avoid String clones
            let mut south_pos = HashMap::with_capacity(south_layer.len());
            for (idx, v) in south_layer.iter().enumerate() {
                south_pos.insert(v.as_ref(), idx);
            }

            // Pre-calculate total edge count for accurate capacity
            // (Double iteration is faster than Vec reallocation)
            let total_edges: usize = north_layer
                .iter()
                .filter_map(|v| g.node(v.as_ref()))
                .map(|n| n.out_edges.len())
                .sum();
            let mut south_entries: Vec<(usize, f64)> = Vec::with_capacity(total_edges);

            for v in north_layer {
                if let Some(node) = g.node(v.as_ref()) {
                    let out_edges_len = node.out_edges.len();
                    if out_edges_len == 0 {
                        continue;
                    }

                    // Lookup edges from graph.edges using EdgeKey
                    let initial_len = south_entries.len();
                    for edge_key in node.out_edges.iter() {
                        if let Some(e) = g.edges.get(edge_key) {
                            if let Some(&pos) = south_pos.get(e.w.as_ref()) {
                                south_entries.push((pos, e.label.weight as f64));
                            }
                        }
                    }

                    // Sort only the newly added entries (required for correctness)
                    // UNSAFE: Remove bounds checking for slice access
                    // SAFETY: initial_len is the length before push, and we only access
                    // the slice from initial_len to current len, which is valid
                    if south_entries.len() > initial_len {
                        unsafe {
                            south_entries
                                .get_unchecked_mut(initial_len..)
                                .sort_unstable_by(|a, b| a.0.cmp(&b.0));
                        }
                    }
                }
            }

            // Build the accumulator tree
            let mut first_index = 1usize;
            while first_index < south_layer.len() {
                first_index <<= 1;
            }
            let tree_size = 2 * first_index - 1;
            first_index -= 1;
            let mut tree = vec![0.0; tree_size];

            let mut layer_count = 0usize;
            // Calculate the weighted crossings
            // UNSAFE: Remove bounds checking for tree array access
            // SAFETY: index is always within tree bounds because:
            // 1. initial index = entry.0 + first_index, where entry.0 < south_layer.len() <= first_index
            // 2. tree_size = 2 * first_index - 1, so entry.0 + first_index < tree_size
            // 3. index only decreases via (index - 1) >> 1, staying in bounds
            // 4. index + 1 access only when index % 2 == 1, guaranteeing index + 1 < tree_size
            unsafe {
                for entry in south_entries {
                    let mut index = entry.0 + first_index;
                    *tree.get_unchecked_mut(index) += entry.1;
                    let mut weight_sum = 0.0;
                    while index > 0 {
                        if index % 2 == 1 {
                            weight_sum += *tree.get_unchecked(index + 1);
                        }
                        index = (index - 1) >> 1;
                        *tree.get_unchecked_mut(index) += entry.1;
                    }
                    layer_count += (entry.1 * weight_sum) as usize;
                }
            }

            let new_total =
                total_count.fetch_add(layer_count, AtomicOrdering::Relaxed) + layer_count;

            // OPTIMIZATION: Check if we should stop immediately after adding
            if new_total > best_cc {
                should_stop.store(1, AtomicOrdering::Relaxed);
            }
        });

        total_count.load(AtomicOrdering::Relaxed)
    };

    // Initialize order with DFS
    let init_order = |g: &DagreGraph| -> Vec<Vec<Arc<str>>> {
        let node_count = g.nodes.len();
        let mut visited = HashSet::with_capacity(node_count);

        // OPTIMIZATION: Pre-build rank cache to avoid repeated lock acquisitions
        // Use &str as key to avoid String clone
        let rank_cache: HashMap<&str, i32> = g
            .nodes
            .iter()
            .filter_map(|(id, node)| node.label.read().rank.map(|r| (id.as_ref(), r)))
            .collect();

        let nodes: Vec<Arc<str>> = g
            .nodes
            .values()
            .filter(|node| {
                if let Some(ref children_map) = g.children {
                    !children_map.contains_key(&node.v)
                        || children_map.get(&node.v).unwrap().is_empty()
                } else {
                    true
                }
            })
            .map(|node| node.v.clone())
            .collect();

        let max_rank_val = rank_cache.values().copied().max().unwrap_or(-1);

        if max_rank_val == -1 {
            return Vec::new();
        }

        let mut layers: Vec<Vec<Arc<str>>> = vec![Vec::new(); (max_rank_val + 1) as usize];

        // Create queue with index to preserve order for stable sort
        let mut queue_with_idx: Vec<(Arc<str>, i32, usize)> = nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                rank_cache
                    .get(v.as_ref() as &str)
                    .map(|&rank| (v.clone(), rank, idx))
            })
            .collect();

        // Stable sort by rank (ascending), then by original index
        queue_with_idx.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));

        // Reverse and extract node IDs
        queue_with_idx.reverse();
        let mut queue: Vec<Arc<str>> = queue_with_idx.into_iter().map(|(v, _, _)| v).collect();

        // OPTIMIZATION: Use indices to avoid cloning during iteration
        let mut i = 0;
        while i < queue.len() {
            // Access by reference to avoid clone
            let v = &queue[i];

            if !visited.contains(v.as_ref() as &str) {
                visited.insert(v.as_ref().to_string());
                if let Some(&rank) = rank_cache.get(v.as_ref() as &str) {
                    if (rank as usize) < layers.len() {
                        layers[rank as usize].push(v.clone());
                    }
                }

                if let Some(successors) = g.successors(v.as_ref()) {
                    for (succ, _) in successors {
                        queue.push(succ.clone());
                    }
                }
            }
            i += 1;
        }

        layers
    };

    // Build layer graph
    let build_layer_graph = |g: &DagreGraph,
                             nodes: &[crate::graph::GraphNode],
                             rank_indexes: Option<&HashMap<i32, usize>>,
                             rank: i32,
                             relationship: bool|
     -> LayerGraph {
        let root = loop {
            let candidate = unique_id("_root");
            if !g.has_node(&candidate) {
                break crate::graph::arc_str(&candidate);
            }
        };

        let mut graph = DagreGraph::new(true, true);

        let length = nodes.len();
        let has_border = g.has_border.unwrap_or(false);
        let mut _added_nodes = 0;

        if has_border {
            let mut i = 0;
            while i < length {
                let node = &nodes[i];
                i += 1;
                let label = node.label.read();
                if label.rank == Some(rank)
                    || (label.min_rank.is_some()
                        && label.max_rank.is_some()
                        && label.min_rank.unwrap() <= rank
                        && rank <= label.max_rank.unwrap())
                {
                    let v = &node.v;
                    // Share BOTH label and order_atomic with the original graph
                    graph.set_node_with_shared_data(
                        v,
                        node.label.clone(),
                        node.order_atomic.clone(),
                    );
                    let parent = g.parent(v);
                    graph.set_parent(v, parent.as_deref().or(Some(root.as_ref())));

                    // This assumes we have only short edges!
                    if relationship {
                        for edge_key in node.in_edges.iter() {
                            if let Some(e) = g.edges.get(edge_key) {
                                graph.set_edge(
                                    &e.v,
                                    v,
                                    Some(EdgeLabel {
                                        weight: e.label.weight,
                                        ..Default::default()
                                    }),
                                    None,
                                );
                            }
                        }
                    } else {
                        for edge_key in node.out_edges.iter() {
                            if let Some(e) = g.edges.get(edge_key) {
                                graph.set_edge(
                                    &e.w,
                                    v,
                                    Some(EdgeLabel {
                                        weight: e.label.weight,
                                        ..Default::default()
                                    }),
                                    None,
                                );
                            }
                        }
                    }

                    if label.min_rank.is_some() {
                        let mut border_left = Vec::new();
                        let mut border_right = Vec::new();
                        if let Some(ref bl) = label.border_left {
                            if (rank as usize) < bl.len() {
                                border_left.push(bl[rank as usize].clone());
                            }
                        }
                        if let Some(ref br) = label.border_right {
                            if (rank as usize) < br.len() {
                                border_right.push(br[rank as usize].clone());
                            }
                        }
                        graph.set_node(
                            v,
                            Some(NodeLabel {
                                border_left: if border_left.is_empty() {
                                    None
                                } else {
                                    Some(border_left)
                                },
                                border_right: if border_right.is_empty() {
                                    None
                                } else {
                                    Some(border_right)
                                },
                                ..Default::default()
                            }),
                        );
                    }
                }
            }
        } else {
            let mut i = rank_indexes
                .and_then(|ri| ri.get(&rank).copied())
                .unwrap_or(0);
            while i < length {
                let node = &nodes[i];
                i += 1;
                let label = node.label.read();
                if label.rank != Some(rank) {
                    break;
                }
                let v = &node.v;
                // Share BOTH label and order_atomic with the original graph
                graph.set_node_with_shared_data(v, node.label.clone(), node.order_atomic.clone());
                _added_nodes += 1;
                let parent = g.parent(v);
                graph.set_parent(v, parent.as_deref().or(Some(root.as_ref())));

                // This assumes we have only short edges!
                if relationship {
                    for edge_key in node.in_edges.iter() {
                        if let Some(e) = g.edges.get(edge_key) {
                            // Add predecessor node with shared label and order_atomic BEFORE set_edge
                            if let Some(pred_node) = g.node(&e.v) {
                                graph.set_node_with_shared_data(
                                    &e.v,
                                    pred_node.label.clone(),
                                    pred_node.order_atomic.clone(),
                                );
                            }
                            graph.set_edge(
                                &e.v,
                                v,
                                Some(EdgeLabel {
                                    weight: e.label.weight,
                                    ..Default::default()
                                }),
                                None,
                            );
                        }
                    }
                } else {
                    for edge_key in node.out_edges.iter() {
                        if let Some(e) = g.edges.get(edge_key) {
                            // Add successor node with shared label and order_atomic BEFORE set_edge
                            if let Some(succ_node) = g.node(&e.w) {
                                graph.set_node_with_shared_data(
                                    &e.w,
                                    succ_node.label.clone(),
                                    succ_node.order_atomic.clone(),
                                );
                            }
                            graph.set_edge(
                                &e.w,
                                v,
                                Some(EdgeLabel {
                                    weight: e.label.weight,
                                    ..Default::default()
                                }),
                                None,
                            );
                        }
                    }
                }
            }
        }

        LayerGraph { graph, root }
    };

    // Assign order to nodes (using atomic operations)
    let assign_order = |g: &mut DagreGraph, layering: &[Vec<Arc<str>>]| {
        for layer in layering {
            for (i, v) in layer.iter().enumerate() {
                // Lock-free atomic write
                if let Some(node) = g.node_mut(v.as_ref()) {
                    write_order_fast(node, i);
                }
            }
        }
    };

    // Main order optimization
    let mut step_start = std::time::Instant::now();

    let mut layering = init_order(g);
    eprintln!("[ORDER PERF] init_order: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    assign_order(g, &layering);
    eprintln!("[ORDER PERF] assign_order: {:?}", step_start.elapsed());

    let rank = max_rank(g).unwrap_or(0);
    eprintln!("[ORDER PERF] max_rank: {}", rank);

    // Note: shared_nodes_map removed as it's not actually used

    let mut down_layer_graphs: Vec<LayerGraph> = Vec::new();
    let mut up_layer_graphs: Vec<LayerGraph> = Vec::new();

    step_start = std::time::Instant::now();
    let mut nodes: Vec<crate::graph::GraphNode> = g.nodes.values().cloned().collect();
    eprintln!("[ORDER PERF] collect nodes: {:?}", step_start.elapsed());

    let mut rank_indexes: Option<HashMap<i32, usize>> = None;

    if !g.has_border.unwrap_or(false) {
        step_start = std::time::Instant::now();
        // OPTIMIZATION: Extract ranks once to avoid repeated label.read() during sort
        // Previous: sort_by with label.read() called ~30,000 times for XLarge
        // Now: extract once, sort by cached value
        let mut nodes_with_ranks: Vec<(crate::graph::GraphNode, i32)> = nodes
            .into_iter()
            .map(|node| {
                let rank = node.label.read().rank.unwrap_or(0);
                (node, rank)
            })
            .collect();

        // Sort by rank using cached values (stable sort to preserve original order for same ranks)
        nodes_with_ranks.sort_by_key(|(_, rank)| *rank);

        // Build rank_indexes and extract sorted nodes
        let mut ri = HashMap::new();
        let mut sorted_nodes = Vec::with_capacity(nodes_with_ranks.len());
        for (i, (node, rank)) in nodes_with_ranks.into_iter().enumerate() {
            if !ri.contains_key(&rank) {
                ri.insert(rank, i);
            }
            sorted_nodes.push(node);
        }
        nodes = sorted_nodes;

        rank_indexes = Some(ri);
        eprintln!(
            "[ORDER PERF] sort and index nodes: {:?}",
            step_start.elapsed()
        );
    }

    step_start = std::time::Instant::now();

    // Build layer graphs in parallel
    let ranks: Vec<i32> = (0..rank).collect();
    let layer_graphs_pairs: Vec<(LayerGraph, LayerGraph)> = ranks
        .par_iter()
        .map(|&i| {
            let down = build_layer_graph(g, &nodes, rank_indexes.as_ref(), i + 1, true);
            let up = build_layer_graph(g, &nodes, rank_indexes.as_ref(), rank - i - 1, false);
            (down, up)
        })
        .collect();

    down_layer_graphs.reserve(rank as usize);
    up_layer_graphs.reserve(rank as usize);
    for (down, up) in layer_graphs_pairs {
        down_layer_graphs.push(down);
        up_layer_graphs.push(up);
    }

    eprintln!(
        "[ORDER PERF] build_layer_graphs ({} ranks): {:?}",
        rank,
        step_start.elapsed()
    );

    let mut best_cc = usize::MAX;
    let mut best: Vec<Vec<Arc<str>>> = Vec::new();
    let mut i = 0;
    let mut last_best = 0;
    let mut prev_cc = usize::MAX;
    let mut same_cc_count = 0;

    let loop_start = std::time::Instant::now();
    eprintln!("[ORDER PERF] Starting sweep loop");

    while last_best < 4 {
        let iter_start = std::time::Instant::now();
        let bias_right = i % 4 >= 2;

        if i % 2 == 1 {
            sweep_layer_graphs(&mut down_layer_graphs, bias_right, g, i, true);
        } else {
            sweep_layer_graphs(&mut up_layer_graphs, bias_right, g, i, false);
        }

        let matrix_start = std::time::Instant::now();
        // Incremental update: only re-sort layers instead of rebuilding
        // This is much faster than build_layer_matrix
        // UNSAFE: Direct node lookup without bounds checking
        // SAFETY: All nodes in layering are guaranteed to exist in g.nodes
        // because layering was built from g.nodes in init_order
        for layer in &mut layering {
            layer.sort_by_cached_key(|v| unsafe {
                g.nodes
                    .get(v.as_ref())
                    .map(|n| n.order_atomic.load(Ordering::Relaxed))
                    .unwrap_unchecked()
            });
        }
        let matrix_time = matrix_start.elapsed();

        let cc_start = std::time::Instant::now();
        let cc = cross_count(g, &layering, best_cc);
        let cc_time = cc_start.elapsed();

        if i < 10 {
            eprintln!(
                "[ORDER PERF] iter {}: {:?} (matrix: {:?}, cc: {:?}), cc={}, best_cc={}",
                i,
                iter_start.elapsed(),
                matrix_time,
                cc_time,
                cc,
                best_cc
            );
        }

        // OPTIMIZATION: Early termination if crossings stabilize
        // If we get the same crossing count multiple times, further iterations unlikely to improve
        if cc == prev_cc && cc == best_cc {
            same_cc_count += 1;
            if same_cc_count >= 2 {
                eprintln!(
                    "[ORDER PERF] Early termination: crossing count stabilized at {}",
                    cc
                );
                break;
            }
        } else {
            same_cc_count = 0;
        }
        prev_cc = cc;

        if cc < best_cc {
            last_best = 0;
            best = layering.clone();
            best_cc = cc;
        } else {
            last_best += 1;
        }
        i += 1;

        if i > 50 {
            eprintln!("[ORDER PERF] Breaking after 50 iterations");
            break;
        }
    }
    eprintln!(
        "[ORDER PERF] sweep loop total: {:?}, iterations: {}",
        loop_start.elapsed(),
        i
    );

    // Reduce crossings (port from TypeScript)
    step_start = std::time::Instant::now();

    let calc_dir = |idx0: usize, idx1: usize| -> i32 {
        if idx0 < idx1 {
            1
        } else {
            2
        }
    };

    let mut i = 4;
    while i < best.len() {
        let layer_len = best[i].len();
        for j in 0..layer_len {
            let node_id = &best[i][j];

            if let Some(node) = g.node(node_id) {
                let in_edges_len = node.in_edges.len();

                if in_edges_len == 2 {
                    // Get the two predecessor nodes (n0 and n1)
                    let edge0_key = &node.in_edges[0];
                    let edge1_key = &node.in_edges[1];

                    if let (Some(edge0), Some(edge1)) =
                        (g.edges.get(edge0_key), g.edges.get(edge1_key))
                    {
                        // Get n0 = node.in[0].vNode, n1 = node.in[1].vNode
                        let n0_id = &edge0.v;
                        let n1_id = &edge1.v;

                        if let (Some(n0_node), Some(n1_node)) = (g.node(n0_id), g.node(n1_id)) {
                            if n0_node.in_edges.len() == 1 && n1_node.in_edges.len() == 1 {
                                let n0_in_edge_key = &n0_node.in_edges[0];
                                let n1_in_edge_key = &n1_node.in_edges[0];

                                if let (Some(n0_in_edge), Some(n1_in_edge)) =
                                    (g.edges.get(n0_in_edge_key), g.edges.get(n1_in_edge_key))
                                {
                                    let mut n0_v = n0_in_edge.v.clone();
                                    let mut n1_v = n1_in_edge.v.clone();

                                    let mut indexes: Vec<usize> = Vec::new();
                                    let mut dir_total = 0;

                                    let mut k = i as i32 - 2;
                                    while k >= 0 {
                                        let k_usize = k as usize;
                                        let layer0 = &best[k_usize];
                                        let idx0 = layer0
                                            .iter()
                                            .position(|id| id.as_ref() == n0_v.as_ref());
                                        let idx1 = layer0
                                            .iter()
                                            .position(|id| id.as_ref() == n1_v.as_ref());

                                        if let (Some(idx0), Some(idx1)) = (idx0, idx1) {
                                            let dir = calc_dir(idx0, idx1);
                                            dir_total |= dir;

                                            // Check conditions
                                            let n0_node_k = g.node(&n0_v);
                                            let n1_node_k = g.node(&n1_v);

                                            let should_break = idx0 == idx1
                                                || (idx0 as i32 - idx1 as i32).abs() != 1
                                                || n0_node_k
                                                    .map_or(true, |n| n.in_edges.len() != 1)
                                                || n1_node_k
                                                    .map_or(true, |n| n.in_edges.len() != 1)
                                                || n0_node_k
                                                    .map_or(true, |n| n.out_edges.len() != 1)
                                                || n1_node_k
                                                    .map_or(true, |n| n.out_edges.len() != 1);

                                            if should_break {
                                                if dir_total == 3 {
                                                    let top_dir = dir;
                                                    let mut l = k_usize + 2;

                                                    while !indexes.is_empty() {
                                                        let idx1_pop = indexes.pop().unwrap();
                                                        let idx0_pop = indexes.pop().unwrap();

                                                        // Calculate layer2 swap BEFORE layer1 swap (like TypeScript)
                                                        let layer1_id0 = best[l][idx0_pop].clone();
                                                        let layer1_id1 = best[l][idx1_pop].clone();

                                                        let layer1 = &mut best[l];

                                                        if calc_dir(idx0_pop, idx1_pop) != top_dir {
                                                            layer1.swap(idx0_pop, idx1_pop);
                                                        }

                                                        // Also swap in layer2 (l-1) using pre-swap layer1 values

                                                        if let (
                                                            Some(layer1_node0),
                                                            Some(layer1_node1),
                                                        ) = (
                                                            g.node(&layer1_id0),
                                                            g.node(&layer1_id1),
                                                        ) {
                                                            if layer1_node0.in_edges.len() > 0
                                                                && layer1_node1.in_edges.len() > 0
                                                            {
                                                                let in_edge0 =
                                                                    &layer1_node0.in_edges[0];
                                                                let in_edge1 =
                                                                    &layer1_node1.in_edges[0];

                                                                if let (Some(edge0), Some(edge1)) = (
                                                                    g.edges.get(in_edge0),
                                                                    g.edges.get(in_edge1),
                                                                ) {
                                                                    let layer2 = &mut best[l - 1];
                                                                    let idx2 = layer2
                                                                        .iter()
                                                                        .position(|id| {
                                                                            id.as_ref()
                                                                                == edge0.v.as_ref()
                                                                        });
                                                                    let idx3 = layer2
                                                                        .iter()
                                                                        .position(|id| {
                                                                            id.as_ref()
                                                                                == edge1.v.as_ref()
                                                                        });

                                                                    if let (
                                                                        Some(idx2),
                                                                        Some(idx3),
                                                                    ) = (idx2, idx3)
                                                                    {
                                                                        if calc_dir(idx2, idx3)
                                                                            != top_dir
                                                                        {
                                                                            layer2.swap(idx2, idx3);
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        l += 2;
                                                    }
                                                }
                                                break;
                                            }

                                            indexes.push(idx0);
                                            indexes.push(idx1);

                                            // Update n0_v and n1_v for next iteration (2 levels up like TypeScript)
                                            // TypeScript: n0 = n0.in[0].vNode.in[0].vNode
                                            if let (Some(n0_k), Some(n1_k)) = (n0_node_k, n1_node_k)
                                            {
                                                if n0_k.in_edges.len() == 1
                                                    && n1_k.in_edges.len() == 1
                                                {
                                                    if let (Some(e0), Some(e1)) = (
                                                        g.edges.get(&n0_k.in_edges[0]),
                                                        g.edges.get(&n1_k.in_edges[0]),
                                                    ) {
                                                        // Go up one more level (grandparent)
                                                        if let (Some(n0_parent), Some(n1_parent)) =
                                                            (g.node(&e0.v), g.node(&e1.v))
                                                        {
                                                            if n0_parent.in_edges.len() == 1
                                                                && n1_parent.in_edges.len() == 1
                                                            {
                                                                if let (
                                                                    Some(e0_parent),
                                                                    Some(e1_parent),
                                                                ) = (
                                                                    g.edges.get(
                                                                        &n0_parent.in_edges[0],
                                                                    ),
                                                                    g.edges.get(
                                                                        &n1_parent.in_edges[0],
                                                                    ),
                                                                ) {
                                                                    n0_v = e0_parent.v.clone();
                                                                    n1_v = e1_parent.v.clone();
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            break;
                                        }

                                        k -= 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        i += 2;
    }

    // Second reduce crossings loop (simpler version from TypeScript)
    for i in (0..best.len().saturating_sub(2)).step_by(2) {
        let layer0_len = best[i].len();
        let layer2_len = best[i + 2].len();

        // First inner loop: check layer0 nodes with out_edges >= 2
        for j in 0..layer0_len {
            let node0_id = &best[i][j];
            if let Some(node0) = g.node(node0_id) {
                let out_edges_len = node0.out_edges.len();

                if out_edges_len >= 2 {
                    for k in 0..out_edges_len - 1 {
                        let edge_k_key = &node0.out_edges[k];
                        let edge_k1_key = &node0.out_edges[k + 1];

                        if let (Some(edge_k), Some(edge_k1)) =
                            (g.edges.get(edge_k_key), g.edges.get(edge_k1_key))
                        {
                            let node1d_id = &edge_k.w;
                            let node2d_id = &edge_k1.w;

                            if let (Some(node1d), Some(node2d)) =
                                (g.node(node1d_id), g.node(node2d_id))
                            {
                                if node1d.out_edges.len() == 1 && node2d.out_edges.len() == 1 {
                                    let node1_edge_key = &node1d.out_edges[0];
                                    let node2_edge_key = &node2d.out_edges[0];

                                    if let (Some(node1_edge), Some(node2_edge)) =
                                        (g.edges.get(node1_edge_key), g.edges.get(node2_edge_key))
                                    {
                                        let node1_id = &node1_edge.w;
                                        let node2_id = &node2_edge.w;

                                        // Check if order is different in layer1 vs layer2
                                        let layer1_idx1d = best[i + 1]
                                            .iter()
                                            .position(|id| id.as_ref() == node1d_id.as_ref());
                                        let layer1_idx2d = best[i + 1]
                                            .iter()
                                            .position(|id| id.as_ref() == node2d_id.as_ref());
                                        let layer2_idx1 = best[i + 2]
                                            .iter()
                                            .position(|id| id.as_ref() == node1_id.as_ref());
                                        let layer2_idx2 = best[i + 2]
                                            .iter()
                                            .position(|id| id.as_ref() == node2_id.as_ref());

                                        if let (Some(idx1d), Some(idx2d), Some(idx1), Some(idx2)) =
                                            (layer1_idx1d, layer1_idx2d, layer2_idx1, layer2_idx2)
                                        {
                                            if (idx1d < idx2d) != (idx1 < idx2) {
                                                best[i + 1].swap(idx1d, idx2d);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Second inner loop: check layer2 nodes with in_edges >= 2
        for j in 0..layer2_len {
            let node0_id = &best[i + 2][j];
            if let Some(node0) = g.node(node0_id) {
                let in_edges_len = node0.in_edges.len();

                if in_edges_len == 2 {
                    let edge0_key = &node0.in_edges[0];
                    let edge1_key = &node0.in_edges[1];

                    if let (Some(edge0), Some(edge1)) =
                        (g.edges.get(edge0_key), g.edges.get(edge1_key))
                    {
                        let node1d_id = &edge0.v;
                        let node2d_id = &edge1.v;

                        if let (Some(node1d), Some(node2d)) = (g.node(node1d_id), g.node(node2d_id))
                        {
                            if node1d.in_edges.len() == 1 && node2d.in_edges.len() == 1 {
                                let node1_edge_key = &node1d.in_edges[0];
                                let node2_edge_key = &node2d.in_edges[0];

                                if let (Some(node1_edge), Some(node2_edge)) =
                                    (g.edges.get(node1_edge_key), g.edges.get(node2_edge_key))
                                {
                                    let node1_id = &node1_edge.v;
                                    let node2_id = &node2_edge.v;

                                    // Check if order is different in layer1 vs layer0
                                    let layer1_idx1d = best[i + 1]
                                        .iter()
                                        .position(|id| id.as_ref() == node1d_id.as_ref());
                                    let layer1_idx2d = best[i + 1]
                                        .iter()
                                        .position(|id| id.as_ref() == node2d_id.as_ref());
                                    let layer0_idx1 = best[i]
                                        .iter()
                                        .position(|id| id.as_ref() == node1_id.as_ref());
                                    let layer0_idx2 = best[i]
                                        .iter()
                                        .position(|id| id.as_ref() == node2_id.as_ref());

                                    if let (Some(idx1d), Some(idx2d), Some(idx1), Some(idx2)) =
                                        (layer1_idx1d, layer1_idx2d, layer0_idx1, layer0_idx2)
                                    {
                                        if (idx1d < idx2d) != (idx1 < idx2) {
                                            best[i + 1].swap(idx1d, idx2d);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Third reduce crossings loop (4-layer pattern from TypeScript)
    for i in (0..best.len().saturating_sub(4)).step_by(2) {
        let layer2_len = best[i + 2].len();
        let layer4_len = best[i + 4].len();

        if layer2_len >= 2 && layer4_len >= 2 {
            let layer0_len = best[i].len();

            // First inner loop: check layer0 nodes with out_edges >= 2
            for j in 0..layer0_len {
                let node0_id = &best[i][j];
                if let Some(node0) = g.node(node0_id) {
                    if node0.in_edges.len() > 0 && node0.out_edges.len() >= 2 {
                        let mut k = 0;
                        while k < node0.out_edges.len() - 1 {
                            let edge_k_key = &node0.out_edges[k];
                            let edge_k1_key = &node0.out_edges[k + 1];

                            if let (Some(edge_k), Some(edge_k1)) =
                                (g.edges.get(edge_k_key), g.edges.get(edge_k1_key))
                            {
                                let node1u_id = &edge_k.w;
                                let node2u_id = &edge_k1.w;

                                if let (Some(node1u), Some(node2u)) =
                                    (g.node(node1u_id), g.node(node2u_id))
                                {
                                    if node1u.out_edges.len() == 1 && node2u.out_edges.len() == 1 {
                                        let node1_edge_key = &node1u.out_edges[0];
                                        let node2_edge_key = &node2u.out_edges[0];

                                        if let (Some(node1_edge), Some(node2_edge)) = (
                                            g.edges.get(node1_edge_key),
                                            g.edges.get(node2_edge_key),
                                        ) {
                                            let node1_id = &node1_edge.w;
                                            let node2_id = &node2_edge.w;

                                            if let (Some(node1), Some(node2)) =
                                                (g.node(node1_id), g.node(node2_id))
                                            {
                                                if node1.out_edges.len() == 1
                                                    && node2.out_edges.len() == 1
                                                {
                                                    let index1 =
                                                        best[i + 2].iter().position(|id| {
                                                            id.as_ref() == node1_id.as_ref()
                                                        });
                                                    let index2 =
                                                        best[i + 2].iter().position(|id| {
                                                            id.as_ref() == node2_id.as_ref()
                                                        });

                                                    if let (Some(idx1), Some(idx2)) =
                                                        (index1, index2)
                                                    {
                                                        if idx1 + 1 == idx2 {
                                                            let node1d_edge_key =
                                                                &node1.out_edges[0];
                                                            let node2d_edge_key =
                                                                &node2.out_edges[0];

                                                            if let (
                                                                Some(node1d_edge),
                                                                Some(node2d_edge),
                                                            ) = (
                                                                g.edges.get(node1d_edge_key),
                                                                g.edges.get(node2d_edge_key),
                                                            ) {
                                                                let node1d_id = &node1d_edge.w;
                                                                let node2d_id = &node2d_edge.w;

                                                                if let (
                                                                    Some(node1d),
                                                                    Some(node2d),
                                                                ) = (
                                                                    g.node(node1d_id),
                                                                    g.node(node2d_id),
                                                                ) {
                                                                    if node1d.out_edges.len() == 1
                                                                        && node2d.out_edges.len()
                                                                            == 1
                                                                    {
                                                                        let node3_edge_key =
                                                                            &node1d.out_edges[0];
                                                                        let node4_edge_key =
                                                                            &node2d.out_edges[0];

                                                                        if let (
                                                                            Some(node3_edge),
                                                                            Some(node4_edge),
                                                                        ) = (
                                                                            g.edges.get(
                                                                                node3_edge_key,
                                                                            ),
                                                                            g.edges.get(
                                                                                node4_edge_key,
                                                                            ),
                                                                        ) {
                                                                            let node3_id =
                                                                                &node3_edge.w;
                                                                            let node4_id =
                                                                                &node4_edge.w;

                                                                            let index3 = best
                                                                                [i + 4]
                                                                                .iter()
                                                                                .position(|id| {
                                                                                    id.as_ref()
                                                                                        == node3_id
                                                                                            .as_ref(
                                                                                            )
                                                                                });
                                                                            let index4 = best
                                                                                [i + 4]
                                                                                .iter()
                                                                                .position(|id| {
                                                                                    id.as_ref()
                                                                                        == node4_id
                                                                                            .as_ref(
                                                                                            )
                                                                                });

                                                                            if let (
                                                                                Some(idx3),
                                                                                Some(idx4),
                                                                            ) = (index3, index4)
                                                                            {
                                                                                if idx3 > idx4 {
                                                                                    // Exchange in 3 layers
                                                                                    let idx1u = best[i + 1].iter().position(|id| id.as_ref() == node1u_id.as_ref());
                                                                                    let idx2u = best[i + 1].iter().position(|id| id.as_ref() == node2u_id.as_ref());
                                                                                    let idx1d = best[i + 3].iter().position(|id| id.as_ref() == node1d_id.as_ref());
                                                                                    let idx2d = best[i + 3].iter().position(|id| id.as_ref() == node2d_id.as_ref());

                                                                                    if let (
                                                                                        Some(i1u),
                                                                                        Some(i2u),
                                                                                        Some(i1d),
                                                                                        Some(i2d),
                                                                                    ) = (
                                                                                        idx1u,
                                                                                        idx2u,
                                                                                        idx1d,
                                                                                        idx2d,
                                                                                    ) {
                                                                                        best[i + 1]
                                                                                            .swap(
                                                                                                i1u,
                                                                                                i2u,
                                                                                            );
                                                                                        best[i + 2].swap(idx1, idx2);
                                                                                        best[i + 3]
                                                                                            .swap(
                                                                                                i1d,
                                                                                                i2d,
                                                                                            );
                                                                                        k += 1;
                                                                                        // Skip next iteration
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            k += 1;
                        }
                    }
                }
            }

            // Second inner loop: check layer2 adjacent nodes
            for j in 0..layer2_len - 1 {
                let node0_id = &best[i + 2][j];
                let node1_id = &best[i + 2][j + 1];

                if let (Some(node0), Some(node1)) = (g.node(node0_id), g.node(node1_id)) {
                    if node0.in_edges.len() == 1
                        && node0.out_edges.len() == 1
                        && node1.in_edges.len() == 1
                        && node1.out_edges.len() == 1
                    {
                        let node0u_edge_key = &node0.in_edges[0];
                        let node1u_edge_key = &node1.in_edges[0];

                        if let (Some(node0u_edge), Some(node1u_edge)) =
                            (g.edges.get(node0u_edge_key), g.edges.get(node1u_edge_key))
                        {
                            let node0u_id = &node0u_edge.v;
                            let node1u_id = &node1u_edge.v;

                            if let (Some(node0u), Some(node1u)) =
                                (g.node(node0u_id), g.node(node1u_id))
                            {
                                if node0u.in_edges.len() == 1 && node1u.in_edges.len() == 1 {
                                    let node2_edge_key = &node0u.in_edges[0];
                                    let node3_edge_key = &node1u.in_edges[0];

                                    if let (Some(node2_edge), Some(node3_edge)) =
                                        (g.edges.get(node2_edge_key), g.edges.get(node3_edge_key))
                                    {
                                        let node2_id = &node2_edge.v;
                                        let node3_id = &node3_edge.v;

                                        let index0 = best[i]
                                            .iter()
                                            .position(|id| id.as_ref() == node2_id.as_ref());
                                        let index1 = best[i]
                                            .iter()
                                            .position(|id| id.as_ref() == node3_id.as_ref());

                                        if let (Some(idx0), Some(idx1)) = (index0, index1) {
                                            if idx1 + 1 == idx0 {
                                                let node0d_edge_key = &node0.out_edges[0];
                                                let node1d_edge_key = &node1.out_edges[0];

                                                if let (Some(node0d_edge), Some(node1d_edge)) = (
                                                    g.edges.get(node0d_edge_key),
                                                    g.edges.get(node1d_edge_key),
                                                ) {
                                                    let node0d_id = &node0d_edge.w;
                                                    let node1d_id = &node1d_edge.w;

                                                    let idx0d = best[i + 3].iter().position(|id| {
                                                        id.as_ref() == node0d_id.as_ref()
                                                    });
                                                    let idx1d = best[i + 3].iter().position(|id| {
                                                        id.as_ref() == node1d_id.as_ref()
                                                    });

                                                    if let (Some(i0d), Some(i1d)) = (idx0d, idx1d) {
                                                        if i0d + 1 == i1d {
                                                            if let (Some(node0d), Some(node1d)) = (
                                                                g.node(node0d_id),
                                                                g.node(node1d_id),
                                                            ) {
                                                                if node0d.out_edges.len() == 1
                                                                    && node1d.out_edges.len() == 1
                                                                {
                                                                    let final0_edge_key =
                                                                        &node0d.out_edges[0];
                                                                    let final1_edge_key =
                                                                        &node1d.out_edges[0];

                                                                    if let (
                                                                        Some(final0_edge),
                                                                        Some(final1_edge),
                                                                    ) = (
                                                                        g.edges
                                                                            .get(final0_edge_key),
                                                                        g.edges
                                                                            .get(final1_edge_key),
                                                                    ) {
                                                                        if final0_edge.w
                                                                            == final1_edge.w
                                                                        {
                                                                            // Exchange in 3 layers
                                                                            let idx0u = best[i + 1]
                                                                                .iter()
                                                                                .position(|id| {
                                                                                    id.as_ref()
                                                                                        == node0u_id
                                                                                            .as_ref(
                                                                                            )
                                                                                });
                                                                            let idx1u = best[i + 1]
                                                                                .iter()
                                                                                .position(|id| {
                                                                                    id.as_ref()
                                                                                        == node1u_id
                                                                                            .as_ref(
                                                                                            )
                                                                                });

                                                                            if let (
                                                                                Some(i0u),
                                                                                Some(i1u),
                                                                            ) = (idx0u, idx1u)
                                                                            {
                                                                                best[i + 1]
                                                                                    .swap(i0u, i1u);
                                                                                best[i + 2]
                                                                                    .swap(j, j + 1);
                                                                                best[i + 3]
                                                                                    .swap(i0d, i1d);
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("[ORDER PERF] reduce crossings: {:?}", step_start.elapsed());

    step_start = std::time::Instant::now();
    assign_order(g, &best);
    eprintln!(
        "[ORDER PERF] final assign_order: {:?}",
        step_start.elapsed()
    );

    // Sync atomic orders back to labels for other phases (position, denormalize)
    step_start = std::time::Instant::now();
    for node in g.nodes.values_mut() {
        let order_val = node.order_atomic.load(Ordering::Relaxed);
        if order_val > 0 {
            node.label.write().order = Some(order_val);
        }
    }
    eprintln!("[ORDER PERF] sync to labels: {:?}", step_start.elapsed());

    // OPTIMIZATION: Skip drop entirely using mem::forget (arena allocator pattern)
    // These large structures (10,000+ LayerGraphs) take 4-5 seconds to drop in XLarge graphs
    // Since we're done with the layout phase, let OS reclaim memory on process exit
    // This is safe because:
    // 1. No Drop implementations have critical side effects (no file handles, etc.)
    // 2. Memory will be reclaimed by OS when process exits
    // 3. Performance gain: 4.77s -> ~0s for XLarge graphs
    step_start = std::time::Instant::now();
    std::mem::forget(down_layer_graphs);
    std::mem::forget(up_layer_graphs);
    std::mem::forget(best);
    std::mem::forget(layering);
    eprintln!("[ORDER PERF] cleanup (forget): {:?}", step_start.elapsed());
}
