use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::types::{Edge, LayoutConfig, LayoutState, Node};

#[napi(object)]
pub struct NapiLayoutResult {
    pub nodes_json: String,
    pub edges_json: String,
    pub width: f64,
    pub height: f64,
}

#[napi(js_name = "layout")]
pub async fn layout_napi(
    nodes_json: String,
    edges_json: String,
    config_json: Option<String>,
) -> Result<NapiLayoutResult> {
    let mut nodes: Vec<Node> = serde_json::from_str(&nodes_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid nodes JSON: {}", e)))?;

    let mut edges: Vec<Edge> = serde_json::from_str(&edges_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid edges JSON: {}", e)))?;

    let layout_config = if let Some(config_str) = config_json {
        serde_json::from_str(&config_str)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid config JSON: {}", e)))?
    } else {
        LayoutConfig::default()
    };

    let mut state = LayoutState::default();

    crate::layout::layout(&mut nodes, &mut edges, &layout_config, &mut state);

    let nodes_result = serde_json::to_string(&nodes).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to serialize nodes: {}", e),
        )
    })?;

    let edges_result = serde_json::to_string(&edges).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to serialize edges: {}", e),
        )
    })?;

    Ok(NapiLayoutResult {
        nodes_json: nodes_result,
        edges_json: edges_result,
        width: state.width,
        height: state.height,
    })
}

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[napi(js_name = "layoutSimple")]
pub async fn layout_simple(input_json: String) -> Result<String> {
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

    let mut input: Input = serde_json::from_str(&input_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid input JSON: {}", e)))?;

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

    serde_json::to_string(&output).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Failed to serialize output: {}", e),
        )
    })
}
