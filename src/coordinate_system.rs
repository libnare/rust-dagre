use crate::graph::DagreGraph;
use crate::types::{LayoutConfig, LayoutState, RankDirection};

pub fn coordinate_system_adjust(g: &mut DagreGraph, layout: &LayoutConfig) {
    let rankdir = layout.rankdir.unwrap_or(RankDirection::TB);

    match rankdir {
        RankDirection::LR => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let (Some(x), Some(y)) = (label.x, label.y) {
                    label.x = Some(y);
                    label.y = Some(x);
                }
                let temp = label.width;
                label.width = label.height;
                label.height = temp;
            }
        }
        RankDirection::RL => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let (Some(x), Some(y)) = (label.x, label.y) {
                    label.x = Some(-y);
                    label.y = Some(x);
                }
                let temp = label.width;
                label.width = label.height;
                label.height = temp;
            }
        }
        RankDirection::BT => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let Some(y) = label.y {
                    label.y = Some(-y);
                }
            }
        }
        RankDirection::TB => {}
    }
}

pub fn coordinate_system_undo(g: &mut DagreGraph, layout: &LayoutConfig) {
    let rankdir = layout.rankdir.unwrap_or(RankDirection::TB);

    match rankdir {
        RankDirection::LR => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let (Some(x), Some(y)) = (label.x, label.y) {
                    label.x = Some(y);
                    label.y = Some(x);
                }
                let temp = label.width;
                label.width = label.height;
                label.height = temp;
            }
        }
        RankDirection::RL => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let (Some(x), Some(y)) = (label.x, label.y) {
                    label.x = Some(-y);
                    label.y = Some(x);
                }
                let temp = label.width;
                label.width = label.height;
                label.height = temp;
            }
        }
        RankDirection::BT => {
            for node in g.nodes.values_mut() {
                let mut label = node.label.write();
                if let Some(y) = label.y {
                    label.y = Some(-y);
                }
            }
        }
        RankDirection::TB => {}
    }
}

pub fn translate_graph(g: &mut DagreGraph, state: &mut LayoutState) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for node in g.nodes.values() {
        if let (Some(x), Some(y)) = (node.label.read().x, node.label.read().y) {
            let half_width = node.label.read().width / 2.0;
            let half_height = node.label.read().height / 2.0;

            min_x = min_x.min(x - half_width);
            max_x = max_x.max(x + half_width);
            min_y = min_y.min(y - half_height);
            max_y = max_y.max(y + half_height);
        }
    }

    // Include edge labels in bounds calculation
    for edge in g.edges.values() {
        if let (Some(x), Some(y), Some(width), Some(height)) = (
            edge.label.x,
            edge.label.y,
            edge.label.width,
            edge.label.height,
        ) {
            let half_width = width / 2.0;
            let half_height = height / 2.0;

            min_x = min_x.min(x - half_width);
            max_x = max_x.max(x + half_width);
            min_y = min_y.min(y - half_height);
            max_y = max_y.max(y + half_height);
        }
    }

    let dx = if min_x != f64::MAX { -min_x } else { 0.0 };
    let dy = if min_y != f64::MAX { -min_y } else { 0.0 };

    for node in g.nodes.values_mut() {
        let mut label = node.label.write();
        if let Some(x) = label.x {
            label.x = Some(x + dx);
        }
        if let Some(y) = label.y {
            label.y = Some(y + dy);
        }
    }

    for edge in g.edges.values_mut() {
        if let Some(ref mut points) = edge.label.points {
            for point in points.iter_mut() {
                point.x += dx;
                point.y += dy;
            }
        }
        if let Some(x) = edge.label.x {
            edge.label.x = Some(x + dx);
        }
        if let Some(y) = edge.label.y {
            edge.label.y = Some(y + dy);
        }
    }

    state.width = if max_x != f64::MIN {
        max_x - min_x
    } else {
        0.0
    };
    state.height = if max_y != f64::MIN {
        max_y - min_y
    } else {
        0.0
    };
}
