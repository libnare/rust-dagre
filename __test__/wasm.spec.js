const fs = require('fs');
const path = require('path');

async function runWasmWrapperTest() {
    console.log("\n--- Testing via Generated JS Wrapper (WASI) ---");
    
    try {
        // The generated CJS wrapper loads and instantiates the WASM module immediately.
        // It handles WASI initialization and thread worker setup.
        const wasmModule = require('../rust-dagre.wasi.cjs');
        console.log("Loaded wasm module wrapper.");

        const nodes = JSON.stringify([
            { v: "1", width: 100, height: 50 },
            { v: "2", width: 100, height: 50 }
        ]);

        const edges = JSON.stringify([
            { v: "1", w: "2" }
        ]);

        if (wasmModule.layout) {
            // layout is now async
            const result = await wasmModule.layout(nodes, edges);
            
            console.log("WASM Result width:", result.width);
            console.log("WASM Result height:", result.height);
            
            // Check camelCase (napi default)
            const nodesResult = result.nodesJson;
            if (nodesResult) {
            console.log("Nodes:", JSON.parse(nodesResult));
            console.log("WASM SUCCESS!");
            } else {
                console.error("nodesJson field missing in result");
            }

        } else {
            console.error("layout function missing from exports");
        }

    } catch (e) {
        console.error("WASM Wrapper Test Failed:", e);
        process.exit(1);
    }
}

runWasmWrapperTest();
