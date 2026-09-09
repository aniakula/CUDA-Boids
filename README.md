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




