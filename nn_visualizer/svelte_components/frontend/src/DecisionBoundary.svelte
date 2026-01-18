<script lang="ts">
  import { onMount, afterUpdate } from 'svelte';

  export let predictions: number[][] = [];
  export let dataPoints: {x: number, y: number, label: number}[] = [];
  export let theme = {
    primaryColor: "#ff4b4b",
    backgroundColor: "#ffffff",
    secondaryBackgroundColor: "#f0f2f6",
    textColor: "#31333F"
  };

  let canvas: HTMLCanvasElement;
  const size = 350;
  const bounds = { min: -1.2, max: 1.2};

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, size, size);

    const resolution = predictions.length || 1;
    const cellSize = size / resolution;

    if (predictions.length > 0) {
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const pred = predictions[j]?.[i] ?? 0.5;

          let r: number, g: number, b: number;
          if (pred < 0.5) {
            const t = pred * 2;
            r = Math.floor(231 + (247 - 231) * t);
            g = Math.floor(76 + (247 - 76) * t);
            b = Math.floor(60 + (247 - 60) * t);
          } else {
            const t = (pred - 0.5) * 2;
            r = Math.floor(247 + (52 - 247) * t);
            g = Math.floor(247 + (152 - 247) * t);
            b = Math.floor(247 + (219 - 247) * t);
          }

          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(i * cellSize, j * cellSize, cellSize + 1, cellSize + 1);
        }
      }
    } else {
      ctx.fillStyle = '#f0f0f0';
      ctx.fillRect(0, 0, size, size);
    }

    ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.lineWidth = 0.5;
    const centerX = size / 2;
    const centerY = size / 2;

    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(size, centerY);
    ctx.stroke();

    dataPoints.forEach((point) => {
      const px = ((point.x - bounds.min) / (bounds.max - bounds.min)) * size;
      const py = ((point.y - bounds.min) / (bounds.max - bounds.min)) * size;

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = point.label === 0 ? '#e74c3c' : '#3498db';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    ctx.fillStyle = theme.textColor;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    // ctx.fillText('-3', 10, size - 5);
    // ctx.fillText('0', centerX, size - 5);
    // ctx.fillText('3', size - 10, size - 5);
    // ctx.textAlign = 'left';
    // ctx.fillText('3', 5, 12);
    // ctx.fillText('0', 5, centerY + 4);
    // ctx.fillText('-3', 5, size - 15);
  }

  onMount(draw);
  afterUpdate(draw);
</script>

<div class="boundary-container">
  <canvas bind:this={canvas} width={size} height={size}></canvas>
<!--  Remove legend for now-->
<!--  <div class="legend">-->
<!--    <div class="legend-item">-->
<!--      <span class="dot orange"></span>-->
<!--      Class 0-->
<!--    </div>-->
<!--    <div class="legend-item">-->
<!--      <span class="dot blue"></span>-->
<!--      Class 1-->
<!--    </div>-->
<!--  </div>-->
</div>

<style>
  .boundary-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  canvas {
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .legend {
    display: flex;
    gap: 1.5rem;
    font-size: 0.8rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .dot.orange {
    background: #e74c3c;
  }

  .dot.blue {
    background: #3498db;
  }
</style>