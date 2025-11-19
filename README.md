# Rust-Dagre

A high-performance Rust port of [dagre.js](https://github.com/dagrejs/dagre), a directed graph layout library.

## Based on
- [dagre](https://github.com/dagrejs/dagre)
- [graphlib](https://github.com/dagrejs/graphlib)

## Overview
This project uses [napi-rs](https://napi.rs) to provide a **Universal Package** supporting both Node.js Native Addons and WebAssembly (WASI).

## Installation

```bash
yarn add @libnare/rust-dagre
# or
npm install @libnare/rust-dagre
```

## Usage

### Node.js

The package automatically loads the native binary for your platform. If the native binary is unavailable, it falls back to the WebAssembly (WASI) version.

**Note**: The `layout` function is asynchronous.

```javascript
const { layout } = require('@libnare/rust-dagre');

async function run() {
  // 1. Define Nodes
  const nodes = JSON.stringify([
    { v: "1", width: 100, height: 50 },
    { v: "2", width: 100, height: 50 }
  ]);

  // 2. Define Edges
  const edges = JSON.stringify([
    { v: "1", w: "2" }
  ]);

  // 3. Optional Configuration
  const config = JSON.stringify({
    rankdir: "TB", // Top-to-Bottom
    align: "ul",   // Up-Left (ul, ur, dl, dr)
    nodesep: 50,
    ranksep: 50,
    edgesep: 10
  });

  // 4. Run Layout
  try {
    const result = await layout(nodes, edges, config);
    
    console.log("Graph Dimensions:", result.width, result.height);
    console.log("Nodes:", JSON.parse(result.nodesJson));
    console.log("Edges:", JSON.parse(result.edgesJson));
  } catch (err) {
    console.error("Layout failed:", err);
  }
}

run();
```

### WebAssembly (Browser)

For browser usage, `napi-rs` handles the WASM loading. The usage is identical to Node.js, returning a Promise.

```javascript
import { layout } from '@libnare/rust-dagre';

async function run() {
  // Define inputs (same as Node.js example)
  const nodes = JSON.stringify([
    { v: "1", width: 100, height: 50 },
    { v: "2", width: 100, height: 50 }
  ]);
  const edges = JSON.stringify([
    { v: "1", w: "2" }
  ]);

  // Usage is identical to Node.js
  const result = await layout(nodes, edges);
  console.log(result);
}

run();
```

## API

### `layout(nodesJson, edgesJson, configJson?)`

Computes the layout for the graph. **Returns a Promise.**

- **nodesJson** (`string`): JSON string array of nodes. Each node must have `v` (id), `width`, and `height`.
- **edgesJson** (`string`): JSON string array of edges. Each edge must have `v` (source) and `w` (target).
- **configJson** (`string`, optional): JSON string for layout configuration.
  - `rankdir`: "TB" | "BT" | "LR" | "RL" (default: "TB")
  - `align`: "ul" | "ur" | "dl" | "dr" (default: undefined)
  - `nodesep`: number (default: 50)
  - `ranksep`: number (default: 50)
  - `edgesep`: number (default: 20)
  - `ranker`: "network-simplex" | "tight-tree" | "longest-path" (default: "network-simplex")

- **Returns**: `Promise<NapiLayoutResult>` resolving to an object containing:
  - `width`: Total graph width.
  - `height`: Total graph height.
  - `nodesJson`: Updated nodes with `x`, `y` coordinates.
  - `edgesJson`: Updated edges with control points (`points`).

### `layoutSimple(inputJson)`

A convenience function that takes a single JSON object containing nodes, edges, and config. **Synchronous function.**

- **inputJson** (`string`): `{ nodes: [...], edges: [...], layout: {...} }`
- **Returns**: JSON string of the output `{ nodes, edges, width, height }`.

## Development

### Prerequisites

- **Rust**: Stable toolchain (install via [rustup](https://rustup.rs/)).
- **Node.js**: v18+ recommended.
- **Yarn**: v4+ recommended.

### Build Commands

1. **Install Dependencies**:
   ```bash
   yarn install
   ```

2. **Build Native & WASM**:
   ```bash
   yarn build
   ```
   This command builds the native addon for your current OS and the WASM target.

3. **Run Tests**:
   <!-- - **Rust Unit Tests**:
     ```bash
     cargo test
     ``` -->
   - **JS Integration Tests**:
     ```bash
     yarn test
     ```

4. **Lint & Format**:
   ```bash
   yarn lint
   yarn format
   ```

## License

MIT
