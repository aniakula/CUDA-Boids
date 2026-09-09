**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Anirudh Akula
  * [LinkedIn](https://www.linkedin.com/in/anirudh-akula/)
* Tested on: Windows 11, NVIDIA T1000 4096MB (CETS Virtual PC)

## Boids Simulation:
### Simulation initially: (small flocks)
<img width="952" height="558" alt="Boids_init" src="https://github.com/user-attachments/assets/e03de8fd-d785-4034-9793-5c8f09d55f68" />

### Simulation after about 1 min: (large flocks)
<img width="786" height="551" alt="Boids_running" src="https://github.com/user-attachments/assets/134e0606-b76f-4ae0-8a3d-2327e9645888" />

## Implementation Overview

This project implements a GPU-accelerated Boids simulation using CUDA. The simulation models flocking behavior through the three primary rules: **separation**, **alignment**, and **cohesion**.

Two neighbor-search approaches are implemented:

* **Brute Force:** Each boid checks every other boid, resulting in \(O(N^2)\) neighbor comparisons.
* **Uniform Grid:** Boids are spatially partitioned into grid cells, allowing each boid to search only its neighboring cells and significantly reducing the number of distance checks.

The uniform grid implementation includes both **scattered** and **coherent** memory-access approaches. The coherent implementation sorts boids by grid cell and reorders their position and velocity data so that neighboring boids are stored contiguously in memory, improving GPU memory access.

## Performance

The simulation was tested with varying numbers of boids to evaluate the performance improvement of memory coherence.

## Analysis

### **How does changing the number of boids affect performance?**

FPS drops as boid count increases for both implementations. At low N, i anticipate that the simulation is bound by fixed kernel-launch/sort overhead rather than actual work, so FPS is roughly flat. As N grows each boid's neighbor search touches more candidates, and fixed per-frame costs (sort, buffer resets, reorder) scale with N so FPS falls off steadily at higher counts.

<p align="center">
  <img width="1040" height="658" alt="Screenshot 2026-09-09 at 8 12 40 AM" src="https://github.com/user-attachments/assets/d068a02f-9a78-420c-9254-4244e6ab98ae" />
</p>

---

### **How does changing block count and block size affect performance?**

Block size 32 was consistently the slowest across every boid count tested. Since 32 threads is only one warp per block that is too few to hide memory latency. Performance improved moving to 64/128/256/512, with 128, 256, and 512 trading the lead depending on N. 1024 was not the fastest at any N despite having the fewest blocks, suggesting occupancy was saturated by 128-512 and the larger block size added no further benefit.

<p align="center">
  <img width="1386" height="688" alt="Screenshot 2026-09-09 at 8 36 50 AM" src="https://github.com/user-attachments/assets/aca26174-847d-45c7-b4fd-3b6538cc39b1" />
</p>

---

### **Did the coherent uniform grid improve performance? Was this expected?**

No, in fact coherent was slower than scattered at every N tested which wasn't expected. The reorder pass (`kernReorderData`) does a guaranteed full read/write of the position/velocity arrays every frame.  It seems like this is the bottleneck and the reorder's fixed cost outweighed its benefit.

---

### **Did checking 27 vs. 8 cells affect performance? Why?**

Surprisingly, the difference was small. 27-cell was actually slightly faster than 8-cell at several boid counts (e.g. N=4,000: 14,142 vs. 9,150 fps), and only converged to nearly identical values by N=64,000+.The 8-cell version does less total work, but it also adds a per-boid branch to pick which octant to check (vs unbranched 27 cell code). At low-to-moderate N, that branching overhead may have outweighed the savings from skipping empty cells.

<p align="center">
  <img width="1386" height="685" alt="Screenshot 2026-09-09 at 8 52 29 AM" src="https://github.com/user-attachments/assets/c1c1b673-e37a-4b1e-ad18-c8ac02f555a9" />
</p>
