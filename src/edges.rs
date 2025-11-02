use crate::graph::DagreGraph;
use crate::types::{EdgeKey, LayoutConfig, LayoutState, NodeLabel, Point};
use crate::utils::{add_dummy_node, build_layer_matrix};

pub fn make_space_for_edge_labels(
    g: &mut DagreGraph,
    _state: &LayoutState,
    layout: &mut LayoutConfig,
) {
    if let Some(ranksep) = layout.ranksep {
        layout.ranksep = Some(ranksep / 2.0);
    }

    let rankdir = layout.rankdir.as_ref().map(|d| d.as_str()).unwrap_or("TB");

    let edges_to_update: Vec<(
        String,
        String,
        i32,
        Option<f64>,
        Option<f64>,
        Option<String>,
    )> = g
        .edges
        .values()
        .map(|e| {
            let minlen = e.label.minlen * 2;
            let mut width = e.label.width;
            let mut height = e.label.height;

            if let Some(labelpos) = &e.label.labelpos {
                if labelpos.to_lowercase() != "c" {
                    let labeloffset = e.label.labeloffset.unwrap_or(0.0);
                    if rankdir == "TB" || rankdir == "BT" {
                        if let Some(w) = width {
                            width = Some(w + labeloffset);
                        }
                    } else {
                        if let Some(h) = height {
                            height = Some(h + labeloffset);
                        }
                    }
                }
            }

            (
                e.v.as_ref().to_string(),
                e.w.as_ref().to_string(),
                minlen,
                width,
                height,
                e.label.labelpos.clone(),
            )
        })
        .collect();

    for (v, w, minlen, width, height, labelpos) in edges_to_update {
        if let Some(edge) = g.edge_mut(&v, &w) {
            edge.label.minlen = minlen;
            edge.label.width = width;
            edge.label.height = height;
            edge.label.labelpos = labelpos;
        }
    }
}

pub fn remove_self_edges(g: &mut DagreGraph) {
    let self_edges: Vec<(String, EdgeKey)> = g
        .edges
        .values()
        .filter(|e| e.v == e.w)
        .map(|e| (e.v.as_ref().to_string(), e.key.clone()))
        .collect();

    for (v, key) in self_edges {
        if let Some(edge) = g.edge_by_key(&key).cloned() {
            if let Some(node) = g.node_mut(&v) {
                node.label
                    .write()
                    .self_edges
                    .get_or_insert_with(Vec::new)
                    .push(crate::types::SelfEdge {
                        e: key.clone(),
                        label: edge.label.clone(),
                    });
            }
            g.remove_edge_by_key(&key);
        }
    }
}

pub fn insert_self_edges(g: &mut DagreGraph) {
    // OPTIMIZATION: Collect self-edge data into HashMap for O(1) lookup
    use ahash::AHashMap as HashMap;
    let self_edge_map: HashMap<String, (Option<i32>, Vec<crate::types::SelfEdge>)> = g
        .nodes
        .values()
        .filter_map(|node| {
            let label = node.label.read();
            let rank = label.rank;
            let self_edges = label.self_edges.clone();
            drop(label);

            self_edges.map(|edges| (node.v.as_ref().to_string(), (rank, edges)))
        })
        .collect();

    // Early return if no self-edges
    if self_edge_map.is_empty() {
        return;
    }

    let layers = build_layer_matrix(g);

    for layer in layers {
        let mut order_shift = 0;

        for (i, v) in layer.iter().enumerate() {
            if let Some(node) = g.node_mut(v) {
                node.label.write().order = Some(i + order_shift);
            }

            // O(1) HashMap lookup instead of O(n) Vec::find
            if let Some((rank, self_edges)) = self_edge_map.get(v.as_ref()) {
                for self_edge in self_edges {
                    order_shift += 1;

                    let mut label = crate::types::NodeLabel::new(
                        self_edge.label.width.unwrap_or(0.0),
                        self_edge.label.height.unwrap_or(0.0),
                    );
                    label.rank = *rank;
                    label.order = Some(i + order_shift);
                    label.e = Some(self_edge.e.clone());
                    label.label = Some(Box::new(self_edge.label.clone()));

                    add_dummy_node(g, "selfedge", label, "_se");
                }

                if let Some(node) = g.node_mut(v) {
                    node.label.write().self_edges = None;
                }
            }
        }
    }
}

pub fn position_self_edges(g: &mut DagreGraph) {
    let dummy_nodes: Vec<String> = g
        .nodes
        .values()
        .filter(|node| node.label.read().dummy.as_deref() == Some("selfedge"))
        .map(|node| node.v.as_ref().to_string())
        .collect();

    for v in dummy_nodes {
        let edge_key_v = g
            .node(&v)
            .and_then(|n| n.label.read().e.as_ref().map(|e| e.v.clone()));
        if let Some(edge_v) = edge_key_v {
            let coords = g.node(&edge_v).map(|n| {
                let label = n.label.read();
                (label.x.unwrap_or(0.0), label.y.unwrap_or(0.0))
            });

            if let Some((x, y)) = coords {
                if let Some(node) = g.node_mut(&v) {
                    let mut label = node.label.write();
                    label.x = Some(x);
                    label.y = Some(y);
                }
            }
        }
    }
}

pub fn fixup_edge_label_coords(g: &mut DagreGraph) {
    for edge in g.edges.values_mut() {
        if edge.label.x.is_some() {
            let labelpos = edge.label.labelpos.as_deref().unwrap_or("r");
            let labeloffset = edge.label.labeloffset.unwrap_or(0.0);

            if labelpos == "l" || labelpos == "r" {
                if let Some(width) = edge.label.width {
                    edge.label.width = Some(width - labeloffset);
                }
            }

            match labelpos {
                "l" => {
                    if let (Some(x), Some(width)) = (edge.label.x, edge.label.width) {
                        edge.label.x = Some(x - width / 2.0 - labeloffset);
                    }
                }
                "r" => {
                    if let (Some(x), Some(width)) = (edge.label.x, edge.label.width) {
                        edge.label.x = Some(x + width / 2.0 + labeloffset);
                    }
                }
                _ => {
                    // Unsupported label position, but don't error
                }
            }
        }

        if edge.label.points.is_none() {
            edge.label.points = Some(Vec::new());
        }
    }
}

// Finds where a line starting at point ({x, y}) would intersect a rectangle
// ({x, y, width, height}) if it were pointing at the rectangle's center.
fn intersect_rect(rect: &NodeLabel, point: &Point) -> Point {
    let x = rect.x.unwrap_or(0.0);
    let y = rect.y.unwrap_or(0.0);

    // Rectangle intersection algorithm from: http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
    let dx = point.x - x;
    let dy = point.y - y;

    let w = rect.width / 2.0;
    let h = rect.height / 2.0;

    if dx == 0.0 && dy == 0.0 {
        // Not possible to find intersection inside of the rectangle, return center
        return Point { x, y };
    }

    if dy.abs() * w > dx.abs() * h {
        // Intersection is top or bottom of rect.
        let h_signed = if dy < 0.0 { -h } else { h };
        Point {
            x: x + (h_signed * dx) / dy,
            y: y + h_signed,
        }
    } else {
        // Intersection is left or right of rect.
        let w_signed = if dx < 0.0 { -w } else { w };
        Point {
            x: x + w_signed,
            y: y + (w_signed * dy) / dx,
        }
    }
}

pub fn assign_node_intersects(g: &mut DagreGraph) {
    let edge_data: Vec<(String, String, Option<Vec<Point>>)> = g
        .edges
        .values()
        .map(|e| {
            let points = if e.label.points.is_none() {
                Some(Vec::new())
            } else {
                e.label.points.clone()
            };
            (e.v.as_ref().to_string(), e.w.as_ref().to_string(), points)
        })
        .collect();

    for (v, w, mut points_opt) in edge_data {
        if let (Some(v_node), Some(w_node)) = (g.node(&v), g.node(&w)) {
            let v_label = v_node.label.read();
            let w_label = w_node.label.read();

            let p1_opt = if let Some(ref p) = points_opt {
                if !p.is_empty() {
                    Some(p[0].clone())
                } else {
                    None
                }
            } else {
                None
            };

            let p2_opt = if let Some(ref p) = points_opt {
                if !p.is_empty() {
                    Some(p[p.len() - 1].clone())
                } else {
                    None
                }
            } else {
                None
            };

            // If points is empty or None, use target node center for p1 and source node center for p2
            let p1 = p1_opt.unwrap_or(Point {
                x: w_label.x.unwrap_or(0.0),
                y: w_label.y.unwrap_or(0.0),
            });

            let p2 = p2_opt.unwrap_or(Point {
                x: v_label.x.unwrap_or(0.0),
                y: v_label.y.unwrap_or(0.0),
            });

            let start_point = intersect_rect(&*v_label, &p1);
            let end_point = intersect_rect(&*w_label, &p2);

            let mut new_points = points_opt.unwrap_or_else(Vec::new);
            new_points.insert(0, start_point);
            new_points.push(end_point);

            points_opt = Some(new_points);
        }

        if let Some(edge) = g.edge_mut(&v, &w) {
            edge.label.points = points_opt;
        }
    }
}
