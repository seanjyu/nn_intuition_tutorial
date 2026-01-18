<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';
  import * as d3 from 'd3';

  export let lossHistory: number[] = [];
  export let theme = {
    primaryColor: "#4b6fff",
    backgroundColor: "#d11717",
    secondaryBackgroundColor: "#000000",
    textColor: "#31333F"
  };

  let container: HTMLDivElement;
  const size = 350;

  function draw() {
    if (!container) return;

    d3.select(container).selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = size - margin.left - margin.right;
    const height = size - margin.top - margin.bottom;

    // const svg = d3.select(container)
    //   .append('svg')
    //   .attr('width', size)
    //   .attr('height', size)
    //   .style('background-color', 'transparent')
    //   .append('g')
    //   .attr('transform', `translate(${margin.left},${margin.top})`);

      const svgElement = d3.select(container)
      .append('svg')
      .attr('width', size)
      .attr('height', size)
      .style('background-color', 'transparent');

    const svg = svgElement.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);


    const data = lossHistory.length > 0 ? lossHistory : [0];

    const x = d3.scaleLinear()
      .domain([0, Math.max(data.length - 1, 1)])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data) || 1])
      .nice()
      .range([height, 0]);

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
      .style('fill', theme.textColor);

    svg.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text')
      .style('fill', theme.textColor);

    svg.selectAll('.domain, .tick line')
      .style('stroke', 'rgba(0,0,0,0.2)');

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', theme.textColor)
      .text('Epoch');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -38)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('fill', theme.textColor)
      .text('Loss');

    if (lossHistory.length > 0) {
      const line = d3.line<number>()
        .x((_, i) => x(i))
        .y(d => y(d))
        .curve(d3.curveMonotoneX);

      svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', theme.primaryColor)
        .attr('stroke-width', 2)
        .attr('d', line);

      svg.selectAll('.dot')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', (_, i) => x(i))
        .attr('cy', d => y(d))
        .attr('r', 3)
        .attr('fill', theme.primaryColor);
    }
  }

  onMount(draw);
  afterUpdate(draw);
</script>

<div bind:this={container}
     class="loss-chart"
     style="background-color: transparent;"
></div>

<style>
  .loss-chart {
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    /*background: #fff;*/
  }
</style>