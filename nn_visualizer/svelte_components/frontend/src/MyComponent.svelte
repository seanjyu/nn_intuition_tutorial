<!--<script lang="ts">-->
<!--  import { setStreamlitLifecycle, Streamlit } from "./streamlit";-->
<!--  setStreamlitLifecycle();-->

<!--  // Props from Python (use export let)-->
<!--  export let name: string = "World";-->
<!--  export let hiddenLayers: number = 2;-->
<!--  export let neuronsPerLayer: number = 4;-->

<!--  // Local state-->
<!--  let count: number = 0;-->

<!--  function handleClick() {-->
<!--    count += 1;-->
<!--    // Send value back to Streamlit-->
<!--    Streamlit.setComponentValue(count);-->
<!--  }-->
<!--</script>-->

<!--<div class="container">-->
<!--  <h2>Hello, {name}!</h2>-->

<!--  <div class="info">-->
<!--    <p>🧠 Hidden Layers: <strong>{hiddenLayers}</strong></p>-->
<!--    <p>⚪ Neurons per Layer: <strong>{neuronsPerLayer}</strong></p>-->
<!--  </div>-->

<!--  <div class="counter">-->
<!--    <p>Button clicked: <strong>{count}</strong> times</p>-->
<!--    <button on:click={handleClick}>Click me!</button>-->
<!--  </div>-->
<!--</div>-->

<!--<style>-->
<!--  .container {-->
<!--    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;-->
<!--    padding: 1.5rem;-->
<!--    background: #f8f9fa;-->
<!--    border-radius: 10px;-->
<!--    max-width: 400px;-->
<!--  }-->

<!--  h2 {-->
<!--    margin: 0 0 1rem 0;-->
<!--    color: #2d3142;-->
<!--  }-->

<!--  .info {-->
<!--    background: white;-->
<!--    padding: 1rem;-->
<!--    border-radius: 8px;-->
<!--    margin-bottom: 1rem;-->
<!--  }-->

<!--  .info p {-->
<!--    margin: 0.5rem 0;-->
<!--    color: #666;-->
<!--  }-->

<!--  .info strong {-->
<!--    color: #2d3142;-->
<!--    font-size: 1.2rem;-->
<!--  }-->

<!--  .counter {-->
<!--    background: white;-->
<!--    padding: 1rem;-->
<!--    border-radius: 8px;-->
<!--    text-align: center;-->
<!--  }-->

<!--  .counter p {-->
<!--    margin: 0 0 1rem 0;-->
<!--  }-->

<!--  .counter strong {-->
<!--    color: #9b59b6;-->
<!--    font-size: 1.5rem;-->
<!--  }-->

<!--  button {-->
<!--    background: #9b59b6;-->
<!--    color: white;-->
<!--    border: none;-->
<!--    padding: 0.75rem 2rem;-->
<!--    font-size: 1rem;-->
<!--    border-radius: 6px;-->
<!--    cursor: pointer;-->
<!--    transition: background 0.2s;-->
<!--  }-->

<!--  button:hover {-->
<!--    background: #8e44ad;-->
<!--  }-->
<!--</style>-->

<!-- nn_visualizer/svelte_components/frontend/src/MyComponent.svelte -->
<script lang="ts">
  import { setStreamlitLifecycle } from "./streamlit";
  import NetworkDiagram from "./NetworkDiagram.svelte";

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
    backgroundColor: "#ffffff",
    secondaryBackgroundColor: "#f0f2f6",
    textColor: "#31333F",
    font: "sans-serif"
  };
</script>

<div
  class="container"
  style="
    background-color: {theme.backgroundColor};
    color: {theme.textColor};
    font-family: {theme.font};
  "
>
  <!-- Network Diagram -->
  <div class="diagram">
    <div class = "centered">
      <NetworkDiagram {layerSizes} {weights} {theme} />
    </div>
  </div>
</div>

<style>
  .container {
    padding: 1.5rem;
    border-radius: 10px;
  }

  h2 {
    margin: 0 0 1rem 0;
    font-size: 1.3rem;
  }

  .metrics {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .metric {
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    min-width: 80px;
  }

  .metric .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.7;
  }

  .metric .value {
    font-size: 1.4rem;
    font-weight: 700;
  }

  .diagram {
    display: flex;
    justify-content: center;
  }
</style>