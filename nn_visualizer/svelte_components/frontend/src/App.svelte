<script lang="ts">
  import MyComponent from "./MyComponent.svelte";
  import NetworkDiagram from "./NetworkDiagram.svelte";
  import DecisionBoundary from "./DecisionBoundary.svelte";
  import LossCurve from "./LossCurve.svelte";

  // Router prop
  export let componentType: string = "network";

  // Props for MyComponent (NetworkDiagram)
  export let name: string = "Student";
  export let layerSizes: number[] = [2, 4, 4, 1];
  export let weights: number[][][] = [];
  export let epoch: number = 0;
  export let loss: number = 0;
  export let animate: boolean = true;

  // Props for DecisionBoundary
  export let predictions: number[][] = [];
  export let dataPoints: { x: number; y: number; label: number }[] = [];

  // Props for LossCurve
  export let lossHistory: number[];

  // Shared
  export let theme = {};
</script>

{#if componentType === "network"}
  <NetworkDiagram {name} {layerSizes} {weights} {epoch} {loss} {theme} />
{:else if componentType === "boundary"}
  <DecisionBoundary {predictions} {dataPoints} {theme} />
{:else if componentType === "loss"}
  <LossCurve {lossHistory} {theme}/>
{/if}