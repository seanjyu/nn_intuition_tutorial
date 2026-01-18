<script lang="ts">
  import { setStreamlitLifecycle } from "./streamlit";
  import { onMount, afterUpdate } from 'svelte';
  import * as d3 from 'd3';

  setStreamlitLifecycle();

  // Props from Python
  export let name: string = "Student";
  export let layerSizes: number[] = [2, 4, 4, 1];
  export let weights: number[][][] = [];
  export let epoch: number = 0;
  export let loss: number = 0;

  // Streamlit theme
  export let theme = {
    base: "light",
    primaryColor: "#ff4b4b",
    backgroundColor: "#471010",
    secondaryBackgroundColor: "#525a70",
    textColor: "#31333F",
    font: "sans-serif"
  };

  let svgElement: SVGSVGElement;

  // const width = 500;
  // const height = 300;
  let width = 500;
  let height = 300;
  let container: HTMLDivElement;
  const margin = { top: 40, right: 30, bottom: 20, left: 30 };

  const colors = {
    input: "#2ecc71",
    hidden: "#9b59b6",
    output: "#e74c3c",
    posWeight: "#3498db",
    negWeight: "#e67e22"
  };

  interface Node {
    x: number;
    y: number;
    layer: number;
    index: number;
  }

  function draw() {
    if (!svgElement) return;

    const svg = d3.select(svgElement);

    // Fade out existing content
    svg.selectAll("g.content")
      .transition()
      .duration(200)
      .attr("opacity", 0)
      .remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create new content group (starts invisible)
    const g = svg.append("g")
      .attr("class", "content")
      .attr("transform", `translate(${margin.left}, ${margin.top})`)
      .attr("opacity", 0);

    const numLayers = layerSizes.length;

    const xScale = d3.scaleLinear()
      .domain([0, numLayers - 1])
      .range([0, innerWidth]);

    // Build nodes
    const nodes: Node[] = [];
    layerSizes.forEach((size, layerIdx) => {
      const yStep = innerHeight / (size + 1);
      for (let i = 0; i < size; i++) {
        nodes.push({
          x: xScale(layerIdx),
          y: yStep * (i + 1),
          layer: layerIdx,
          index: i
        });
      }
    });

    // Draw connections
    let nodeOffset = 0;
    for (let l = 0; l < numLayers - 1; l++) {
      const sourceStart = nodeOffset;
      nodeOffset += layerSizes[l];
      const targetStart = nodeOffset;
      const W = weights[l] || [];

      for (let i = 0; i < layerSizes[l]; i++) {
        for (let j = 0; j < layerSizes[l + 1]; j++) {
          const sourceNode = nodes[sourceStart + i];
          const targetNode = nodes[targetStart + j];
          const w = W[j]?.[i] ?? 0;
          const wNorm = Math.tanh(w);
          const color = wNorm >= 0 ? colors.posWeight : colors.negWeight;
          const opacity = Math.abs(wNorm) * 0.6 + 0.15;
          const strokeWidth = Math.abs(wNorm) * 2.5 + 0.5;

          g.append("line")
            .attr("x1", sourceNode.x)
            .attr("y1", sourceNode.y)
            .attr("x2", targetNode.x)
            .attr("y2", targetNode.y)
            .attr("stroke", color)
            .attr("stroke-width", strokeWidth)
            .attr("stroke-opacity", opacity);
        }
      }
    }

    // Draw nodes
    nodes.forEach((node) => {
      const isInput = node.layer === 0;
      const isOutput = node.layer === numLayers - 1;

      let color: string;
      if (isInput) color = colors.input;
      else if (isOutput) color = colors.output;
      else color = colors.hidden;

      g.append("circle")
        .attr("cx", node.x)
        .attr("cy", node.y)
        .attr("r", 14)
        .attr("fill", color)
        .attr("stroke", "#fff")
        .attr("stroke-width", 2.5);
    });

    // Layer labels
    const labels = [
      "Input",
      ...Array(numLayers - 2).fill(0).map((_, i) => `Hidden ${i + 1}`),
      "Output"
    ];

    labels.forEach((label, i) => {
      g.append("text")
        .attr("x", xScale(i))
        .attr("y", -15)
        .attr("text-anchor", "middle")
        .attr("font-size", "12px")
        .attr("font-weight", "500")
        .attr("fill", theme.textColor)
        .text(label);
    });

    // Fade in new content
    g.transition()
      .duration(300)
      .attr("opacity", 1);
  }

  // onMount(draw);
    onMount(() => {
    const resizeObserver = new ResizeObserver(() => {
      if (container) {
        width = Math.min(container.clientWidth - 30, 800); // responsive width
        height = Math.min(container.clientHeight, 500);
        draw();
      }
    });

    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  });
  afterUpdate(draw);
</script>

<div
  bind:this={container}
  class="container"
  style="
    background-color: {theme.backgroundColor};
    color: {theme.textColor};
    font-family: {theme.font};
  "
>
  <div class="diagram">
    <div class="centered">
      <svg
        bind:this={svgElement}
        {width}
        {height}
        style="background-color: transparent;"
      ></svg>
    </div>
  </div>
</div>

<style>
  .container {
    padding: 1.5rem;
    border-radius: 10px;
  }

  .diagram {
    display: flex;
    justify-content: center;
  }

  svg {
    display: block;
    margin: 0 auto;
  }
</style>