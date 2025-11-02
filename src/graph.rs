use crate::types::{EdgeKey, EdgeLabel, NodeLabel};
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use indexmap::{IndexMap, IndexSet};
use parking_lot::RwLock;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

const NULL_PARENT: &str = "\x00";

// Helper function to convert &str to Arc<str>
#[inline]
pub fn arc_str(s: &str) -> Arc<str> {
    Arc::from(s)
}

// Helper function to convert Arc<str> to String
#[inline]
pub fn arc_to_string(s: &Arc<str>) -> String {
    s.as_ref().to_string()
}

#[derive(Clone)]
pub struct GraphNode {
    pub label: Arc<RwLock<NodeLabel>>,
    // Separate atomic order for lock-free access during order phase
    pub order_atomic: Arc<AtomicUsize>,
    // Arc-wrapped Vecs for cheap cloning (critical for performance!)
    pub in_edges: Arc<Vec<EdgeKey>>,
    pub out_edges: Arc<Vec<EdgeKey>>,
    pub predecessors: HashMap<Arc<str>, usize>,
    pub successors: HashMap<Arc<str>, usize>,
    pub v: Arc<str>,
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("label", &*self.label.read())
            .field("in_edges", &self.in_edges)
            .field("out_edges", &self.out_edges)
            .field("predecessors", &self.predecessors)
            .field("successors", &self.successors)
            .field("v", &self.v)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub label: EdgeLabel,
    pub v: Arc<str>,
    pub w: Arc<str>,
    pub name: Option<Arc<str>>,
    pub key: EdgeKey,
}

pub struct DagreGraph {
    pub directed: bool,
    pub compound: bool,
    pub nodes: IndexMap<Arc<str>, GraphNode>,
    pub edges: IndexMap<EdgeKey, GraphEdge>,
    pub parent: Option<HashMap<Arc<str>, Arc<str>>>,
    pub children: Option<HashMap<Arc<str>, IndexSet<Arc<str>>>>,
    pub default_node_label_fn: Option<Box<dyn Fn(&str) -> Option<NodeLabel> + Send + Sync>>,
    pub has_border: Option<bool>,
    pub root: Option<Arc<str>>,
    // Nodes marked as logically removed (for denormalize performance)
    pub removed_nodes: HashSet<Arc<str>>,
}

impl std::fmt::Debug for DagreGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DagreGraph")
            .field("directed", &self.directed)
            .field("compound", &self.compound)
            .field("nodes", &self.nodes)
            .field("edges", &self.edges)
            .field("parent", &self.parent)
            .field("children", &self.children)
            .field(
                "default_node_label_fn",
                &self.default_node_label_fn.as_ref().map(|_| "Fn"),
            )
            .field("has_border", &self.has_border)
            .field("root", &self.root)
            .finish()
    }
}

impl Clone for DagreGraph {
    fn clone(&self) -> Self {
        DagreGraph {
            directed: self.directed,
            compound: self.compound,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            parent: self.parent.clone(),
            children: self.children.clone(),
            default_node_label_fn: None, // Cannot clone function pointers
            has_border: self.has_border,
            root: self.root.clone(),
            removed_nodes: self.removed_nodes.clone(),
        }
    }
}

impl DagreGraph {
    pub fn new(directed: bool, compound: bool) -> Self {
        let mut graph = DagreGraph {
            directed,
            compound,
            nodes: IndexMap::new(),
            edges: IndexMap::new(),
            parent: None,
            children: None,
            default_node_label_fn: Some(Box::new(|_| None)),
            has_border: None,
            root: None,
            removed_nodes: HashSet::new(),
        };

        if compound {
            let parent = HashMap::new();
            let mut children = HashMap::new();
            children.insert(arc_str(NULL_PARENT), IndexSet::new());
            graph.parent = Some(parent);
            graph.children = Some(children);
        }

        graph
    }

    pub fn set_default_node_label(
        &mut self,
        f: Box<dyn Fn(&str) -> Option<NodeLabel> + Send + Sync>,
    ) {
        self.default_node_label_fn = Some(f);
    }

    // Set node with a shared Arc<RwLock<NodeLabel>> and order_atomic
    pub fn set_node_with_label_rc(&mut self, v: &str, label_rc: Arc<RwLock<NodeLabel>>) {
        // For layer graphs, we need to share BOTH label and order_atomic with original graph
        // Check if node exists in original graph to get its order_atomic
        let v_arc = arc_str(v);
        if let Some(node) = self.nodes.get_mut(v_arc.as_ref()) {
            node.label = label_rc;
            // Note: order_atomic is already set from first creation
        } else {
            let node = GraphNode {
                label: label_rc,
                order_atomic: Arc::new(AtomicUsize::new(0)),
                in_edges: Arc::new(Vec::new()),
                out_edges: Arc::new(Vec::new()),
                predecessors: HashMap::new(),
                successors: HashMap::new(),
                v: v_arc.clone(),
            };
            self.nodes.insert(v_arc.clone(), node);

            if self.compound {
                if let Some(ref mut parent) = self.parent {
                    parent.insert(v_arc.clone(), arc_str(NULL_PARENT));
                }
                if let Some(ref mut children) = self.children {
                    children.insert(v_arc, IndexSet::new());
                }
            }
        }
    }

    // Set node with shared label and order_atomic (for layer graphs)
    pub fn set_node_with_shared_data(
        &mut self,
        v: &str,
        label_rc: Arc<RwLock<NodeLabel>>,
        order_atomic: Arc<AtomicUsize>,
    ) {
        let v_arc = arc_str(v);
        if let Some(node) = self.nodes.get_mut(v_arc.as_ref()) {
            // Always update BOTH label and order_atomic to ensure they're from the same source
            node.label = label_rc;
            node.order_atomic = order_atomic;
        } else {
            let node = GraphNode {
                label: label_rc,
                order_atomic,
                in_edges: Arc::new(Vec::new()),
                out_edges: Arc::new(Vec::new()),
                predecessors: HashMap::new(),
                successors: HashMap::new(),
                v: v_arc.clone(),
            };
            self.nodes.insert(v_arc.clone(), node);

            if self.compound {
                if let Some(ref mut parent) = self.parent {
                    parent.insert(v_arc.clone(), arc_str(NULL_PARENT));
                }
                if let Some(ref mut children) = self.children {
                    children.insert(v_arc, IndexSet::new());
                }
            }
        }
    }

    pub fn set_node(&mut self, v: &str, label: Option<NodeLabel>) {
        let v_arc = arc_str(v);
        if let Some(node) = self.nodes.get_mut(v_arc.as_ref()) {
            if let Some(label) = label {
                *node.label.write() = label;
            }
        } else {
            let node_label = if let Some(label) = label {
                label
            } else if let Some(ref f) = self.default_node_label_fn {
                f(v).unwrap_or_else(|| NodeLabel::new(0.0, 0.0))
            } else {
                NodeLabel::new(0.0, 0.0)
            };

            let node = GraphNode {
                label: Arc::new(RwLock::new(node_label)),
                order_atomic: Arc::new(AtomicUsize::new(0)),
                in_edges: Arc::new(Vec::new()),
                out_edges: Arc::new(Vec::new()),
                predecessors: HashMap::new(),
                successors: HashMap::new(),
                v: v_arc.clone(),
            };

            self.nodes.insert(v_arc.clone(), node);

            if self.compound {
                if let Some(ref mut parent) = self.parent {
                    parent.insert(v_arc.clone(), arc_str(NULL_PARENT));
                }
                if let Some(ref mut children) = self.children {
                    children.insert(v_arc.clone(), IndexSet::new());
                    children
                        .get_mut(arc_str(NULL_PARENT).as_ref())
                        .unwrap()
                        .insert(v_arc);
                }
            }
        }
    }

    pub fn node(&self, v: &str) -> Option<&GraphNode> {
        if self.removed_nodes.contains(v) {
            return None;
        }
        self.nodes.get(v)
    }

    pub fn node_mut(&mut self, v: &str) -> Option<&mut GraphNode> {
        if self.removed_nodes.contains(v) {
            return None;
        }
        self.nodes.get_mut(v)
    }

    pub fn has_node(&self, v: &str) -> bool {
        !self.removed_nodes.contains(v) && self.nodes.contains_key(v)
    }

    pub fn sort_nodes_by_rank(&mut self) {
        self.nodes.sort_by(|k1, v1, k2, v2| {
            let rank1 = v1.label.read().rank.unwrap_or(0);
            let rank2 = v2.label.read().rank.unwrap_or(0);
            rank1.cmp(&rank2).then_with(|| k1.cmp(k2))
        });
    }

    pub fn remove_nodes_batch(&mut self, nodes: &[String]) {
        let nodes_set: HashSet<&str> = nodes.iter().map(|s| s.as_ref()).collect();

        // Collect affected nodes and edges
        let mut affected_nodes = HashSet::new();
        let mut edges_to_remove = HashSet::new();

        for node_id in nodes {
            if let Some(node) = self.nodes.get(node_id.as_str()) {
                // Collect predecessors and successors
                for pred in node.predecessors.keys() {
                    affected_nodes.insert(pred.clone());
                }
                for succ in node.successors.keys() {
                    affected_nodes.insert(succ.clone());
                }

                // Collect edges
                for edge_key in node.in_edges.iter() {
                    edges_to_remove.insert(edge_key.clone());
                }
                for edge_key in node.out_edges.iter() {
                    edges_to_remove.insert(edge_key.clone());
                }
            }
        }

        // Remove edges
        for edge_key in &edges_to_remove {
            self.edges.swap_remove(edge_key);
        }

        // Update only affected nodes
        for affected_node_id in affected_nodes {
            if let Some(node) = self.nodes.get_mut(&affected_node_id) {
                node.predecessors
                    .retain(|k, _| !nodes_set.contains(k.as_ref()));
                node.successors
                    .retain(|k, _| !nodes_set.contains(k.as_ref()));
                Arc::make_mut(&mut node.in_edges).retain(|k| !edges_to_remove.contains(k));
                Arc::make_mut(&mut node.out_edges).retain(|k| !edges_to_remove.contains(k));
            }
        }

        // Remove nodes using retain for efficient batch removal
        self.nodes.retain(|k, _| !nodes_set.contains(k.as_ref()));
    }

    pub fn remove_dummy_node(&mut self, v: &str) {
        // Simplified removal for dummy nodes (no compound graph logic)
        let (in_edges, out_edges) = if let Some(node) = self.nodes.get(v) {
            (node.in_edges.clone(), node.out_edges.clone())
        } else {
            return;
        };

        // Remove edges (already done in denormalize, but just in case)
        for edge_key in in_edges.iter() {
            if let Some(edge) = self.edges.shift_remove(edge_key) {
                if let Some(w_node) = self.nodes.get_mut(&edge.w) {
                    w_node.predecessors.retain(|k, _| k.as_ref() != v);
                    Arc::make_mut(&mut w_node.in_edges).retain(|k| k != edge_key);
                }
            }
        }
        for edge_key in out_edges.iter() {
            if let Some(edge) = self.edges.shift_remove(edge_key) {
                if let Some(v_node) = self.nodes.get_mut(&edge.v) {
                    v_node.successors.retain(|k, _| k.as_ref() != v);
                    Arc::make_mut(&mut v_node.out_edges).retain(|k| k != edge_key);
                }
            }
        }

        // Remove node
        self.nodes.shift_remove(v);
    }

    pub fn remove_node(&mut self, v: &str) {
        let (in_edges, out_edges) = if let Some(node) = self.nodes.get(v) {
            (node.in_edges.clone(), node.out_edges.clone())
        } else {
            return;
        };

        if self.compound {
            if let Some(ref mut parent_map) = self.parent {
                if let Some(parent) = parent_map.get(v).cloned() {
                    if let Some(ref mut children) = self.children {
                        if let Some(children_set) = children.get_mut(&parent) {
                            children_set.swap_remove(v);
                        }
                    }
                }
                parent_map.remove(v);
            }

            let children_to_reparent: Vec<Arc<str>> = self.children(Some(v)).collect();
            for child in children_to_reparent {
                self.set_parent(&child.as_ref(), None);
            }

            if let Some(ref mut children) = self.children {
                children.remove(v);
            }
        }

        for edge_key in in_edges.iter() {
            self.remove_edge_by_key(edge_key);
        }
        for edge_key in out_edges.iter() {
            self.remove_edge_by_key(edge_key);
        }

        self.nodes.shift_remove(v);
    }

    pub fn set_parent(&mut self, v: &str, parent: Option<&str>) {
        if !self.compound {
            panic!("Cannot set parent in a non-compound graph");
        }

        let parent_str = parent.unwrap_or(NULL_PARENT);
        let parent_arc = arc_str(parent_str);

        if let Some(parent) = parent {
            let mut ancestor_opt = Some(arc_str(parent));
            while let Some(anc) = ancestor_opt {
                if anc.as_ref() == v {
                    panic!("Setting {} as parent of {} would create a cycle", parent, v);
                }
                ancestor_opt = self.parent(&anc).map(arc_str);
            }

            if !self.has_node(parent) {
                self.set_node(parent, Some(NodeLabel::new(0.0, 0.0)));
            }
        }

        let v_arc = arc_str(v);
        if let Some(ref mut parent_map) = self.parent {
            if let Some(current_parent) = parent_map.get(v_arc.as_ref()).cloned() {
                if let Some(ref mut children) = self.children {
                    if let Some(children_set) = children.get_mut(&current_parent) {
                        children_set.swap_remove(v_arc.as_ref());
                    }
                }
            }
            parent_map.insert(v_arc.clone(), parent_arc.clone());
        }

        if let Some(ref mut children) = self.children {
            children
                .entry(parent_arc)
                .or_insert_with(IndexSet::new)
                .insert(v_arc);
        }
    }

    pub fn parent(&self, v: &str) -> Option<&str> {
        if self.compound {
            if let Some(ref parent_map) = self.parent {
                if let Some(parent) = parent_map.get(v) {
                    if parent.as_ref() != NULL_PARENT {
                        return Some(parent.as_ref());
                    }
                }
            }
        }
        None
    }

    pub fn children(&self, v: Option<&str>) -> Box<dyn Iterator<Item = Arc<str>> + '_> {
        if self.compound {
            let key = v.unwrap_or(NULL_PARENT);
            if let Some(ref children) = self.children {
                if let Some(children_set) = children.get(key) {
                    return Box::new(children_set.iter().cloned());
                }
            }
            Box::new(std::iter::empty())
        } else if v.is_none() {
            Box::new(self.nodes.keys().cloned())
        } else if self.has_node(v.unwrap()) {
            Box::new(std::iter::empty())
        } else {
            Box::new(std::iter::empty())
        }
    }

    pub fn has_children(&self, v: Option<&str>) -> bool {
        if self.compound {
            let key = v.unwrap_or(NULL_PARENT);
            if let Some(ref children) = self.children {
                if let Some(children_set) = children.get(key) {
                    return !children_set.is_empty();
                }
            }
            false
        } else if v.is_none() {
            !self.nodes.is_empty()
        } else {
            false
        }
    }

    pub fn predecessors(&self, v: &str) -> Option<&HashMap<Arc<str>, usize>> {
        self.nodes.get(v).map(|node| &node.predecessors)
    }

    pub fn successors(&self, v: &str) -> Option<&HashMap<Arc<str>, usize>> {
        self.nodes.get(v).map(|node| &node.successors)
    }

    pub fn neighbors(&self, v: &str) -> IndexSet<String> {
        let mut set = IndexSet::new();
        if let Some(node) = self.nodes.get(v) {
            for k in node.predecessors.keys() {
                set.insert(k.as_ref().to_string());
            }
            for k in node.successors.keys() {
                set.insert(k.as_ref().to_string());
            }
        }
        set
    }

    pub fn edge(&self, v: &str, w: &str) -> Option<&GraphEdge> {
        let key = self.edge_key(self.directed, v, w, None);
        self.edges.get(&key)
    }

    pub fn edge_by_key(&self, key: &EdgeKey) -> Option<&GraphEdge> {
        self.edges.get(key)
    }

    pub fn edge_mut(&mut self, v: &str, w: &str) -> Option<&mut GraphEdge> {
        let key = self.edge_key(self.directed, v, w, None);
        self.edges.get_mut(&key)
    }

    pub fn set_edge(&mut self, v: &str, w: &str, label: Option<EdgeLabel>, name: Option<&str>) {
        let key = self.edge_key(self.directed, v, w, name);

        if let Some(edge) = self.edges.get_mut(&key) {
            if let Some(label) = label {
                edge.label = label;
            }
        } else {
            let (v_arc, w_arc) = if !self.directed && v > w {
                (arc_str(w), arc_str(v))
            } else {
                (arc_str(v), arc_str(w))
            };

            let edge_label = label.unwrap_or_else(|| EdgeLabel::new(1.0, 1));

            self.set_node(&v_arc, None);
            self.set_node(&w_arc, None);

            let edge = GraphEdge {
                label: edge_label,
                v: v_arc.clone(),
                w: w_arc.clone(),
                name: name.map(arc_str),
                key: key.clone(),
            };

            self.edges.insert(key.clone(), edge);

            if let Some(w_node) = self.nodes.get_mut(w_arc.as_ref()) {
                *w_node.predecessors.entry(v_arc.clone()).or_insert(0) += 1;
                Arc::make_mut(&mut w_node.in_edges).push(key.clone());
            }

            if let Some(v_node) = self.nodes.get_mut(v_arc.as_ref()) {
                *v_node.successors.entry(w_arc.clone()).or_insert(0) += 1;
                Arc::make_mut(&mut v_node.out_edges).push(key);
            }
        }
    }

    // OPTIMIZATION: set_edge without creating nodes (for normalize where nodes already exist)
    pub fn set_edge_fast(&mut self, v: &str, w: &str, label: EdgeLabel, name: Option<&str>) {
        let key = self.edge_key(self.directed, v, w, name);

        let (v_arc, w_arc) = if !self.directed && v > w {
            (arc_str(w), arc_str(v))
        } else {
            (arc_str(v), arc_str(w))
        };

        let edge = GraphEdge {
            label,
            v: v_arc.clone(),
            w: w_arc.clone(),
            name: name.map(arc_str),
            key: key.clone(),
        };

        self.edges.insert(key.clone(), edge);

        // SAFETY: Caller guarantees nodes exist
        if let Some(w_node) = self.nodes.get_mut(w_arc.as_ref()) {
            *w_node.predecessors.entry(v_arc.clone()).or_insert(0) += 1;
            Arc::make_mut(&mut w_node.in_edges).push(key.clone());
        }

        if let Some(v_node) = self.nodes.get_mut(v_arc.as_ref()) {
            *v_node.successors.entry(w_arc.clone()).or_insert(0) += 1;
            Arc::make_mut(&mut v_node.out_edges).push(key);
        }
    }

    pub fn remove_edge(&mut self, v: &str, w: &str, name: Option<&str>) {
        let key = self.edge_key(self.directed, v, w, name);
        self.remove_edge_by_key(&key);
    }

    pub fn remove_edge_by_key(&mut self, key: &EdgeKey) {
        if let Some(edge) = self.edges.shift_remove(key) {
            let v = &edge.v;
            let w = &edge.w;

            if let Some(w_node) = self.nodes.get_mut(w) {
                if let Some(count) = w_node.predecessors.get_mut(v) {
                    if *count == 1 {
                        w_node.predecessors.remove(v);
                    } else {
                        *count -= 1;
                    }
                }
                Arc::make_mut(&mut w_node.in_edges).retain(|k| k != key);
            }

            if let Some(v_node) = self.nodes.get_mut(v) {
                if let Some(count) = v_node.successors.get_mut(w) {
                    if *count == 1 {
                        v_node.successors.remove(w);
                    } else {
                        *count -= 1;
                    }
                }
                Arc::make_mut(&mut v_node.out_edges).retain(|k| k != key);
            }
        }
    }

    pub(crate) fn edge_key(&self, is_directed: bool, v: &str, w: &str, name: Option<&str>) -> EdgeKey {
        let (v_arc, w_arc) = if !is_directed && v > w {
            (arc_str(w), arc_str(v))
        } else {
            (arc_str(v), arc_str(w))
        };

        EdgeKey::new(v_arc, w_arc, name.map(arc_str))
    }
}
