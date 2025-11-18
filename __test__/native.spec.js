const { layout } = require('../index.js');

const nodes = JSON.stringify([
  { v: "1", width: 100, height: 50 },
  { v: "2", width: 100, height: 50 }
]);

const edges = JSON.stringify([
  { v: "1", w: "2" }
]);

console.log("Testing layout...");

(async () => {
try {
    // Call async layout
    const result = await layout(nodes, edges);
  console.log("Result Object:", result);
  
  console.log("Result width:", result.width);
  console.log("Result height:", result.height);
  
    // Check field names (napi-rs converts snake_case to camelCase in JS object)
    const nodesStr = result.nodesJson;
    const edgesStr = result.edgesJson;
  
  if (nodesStr) console.log("Nodes:", JSON.parse(nodesStr));
    else console.log("nodesJson field is missing");
  
  if (edgesStr) console.log("Edges:", JSON.parse(edgesStr));
    else console.log("edgesJson field is missing");
  
  console.log("SUCCESS!");
} catch (e) {
  console.error("FAILED:", e);
    process.exit(1);
}
})();
