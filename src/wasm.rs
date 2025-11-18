use crate::types::{Edge, LayoutConfig, LayoutState, Node};
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmLayoutResult {
    nodes_json: String,
    edges_json: String,
    width: f64,
    height: f64,
}

#[wasm_bindgen]
impl WasmLayoutResult {
    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> String {
        self.nodes_json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn edges(&self) -> String {
        self.edges_json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f64 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }
}

#[wasm_bindgen]
pub fn layout_wasm(
    nodes_json: &str,
    edges_json: &str,
    config_json: Option<String>,
) -> Result<WasmLayoutResult, JsValue> {
    let mut nodes: Vec<Node> =
        serde_json::from_str(nodes_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut edges: Vec<Edge> =
        serde_json::from_str(edges_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let layout_config = if let Some(config_str) = config_json {
        serde_json::from_str(&config_str).map_err(|e| JsValue::from_str(&e.to_string()))?
    } else {
        LayoutConfig::default()
    };

    let mut state = LayoutState::default();

    crate::layout::layout(&mut nodes, &mut edges, &layout_config, &mut state);

    let nodes_json =
        serde_json::to_string(&nodes).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let edges_json =
        serde_json::to_string(&edges).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmLayoutResult {
        nodes_json,
        edges_json,
        width: state.width,
        height: state.height,
    })
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn layout_simple(input_json: &str) -> Result<String, JsValue> {
    #[derive(serde::Deserialize)]
    struct Input {
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        #[serde(default)]
        layout: LayoutConfig,
    }

    #[derive(serde::Serialize)]
    struct Output {
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        width: f64,
        height: f64,
    }

    let mut input: Input =
        serde_json::from_str(input_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut state = LayoutState::default();

    crate::layout::layout(
        &mut input.nodes,
        &mut input.edges,
        &input.layout,
        &mut state,
    );

    let output = Output {
        nodes: input.nodes,
        edges: input.edges,
        width: state.width,
        height: state.height,
    };

    serde_json::to_string(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}
