#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
//coherent buffers for pos and ping pong velocity
glm::vec3* dev_pos_sorted;
glm::vec3* dev_vel1_sorted;
glm::vec3* dev_vel2_sorted;


// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();

  //each Boid has a grid and particle index:
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));

  //ranges of boids in each grid cell 
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));

  //allocate sorted buffers
  cudaMalloc((void**)&dev_pos_sorted, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_vel1_sorted, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_vel2_sorted, N * sizeof(glm::vec3));

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);

  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/


__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3* pos) {
	glm::vec3 perceived_center(0.0f);
	int num_neighbors = 0;
	int currIdx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (currIdx >= N) { return perceived_center; }
    for(int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule1Distance) {
            perceived_center += pos[i];
			num_neighbors++;
        }
	}
    if (num_neighbors == 0) {
        return glm::vec3(0.0f);
    }
	perceived_center /= num_neighbors;
	return (perceived_center - pos[iSelf]) * rule1Scale;
}

__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3* pos) {
	glm::vec3 seperation_force (0.0f);
    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule2Distance) {
            seperation_force -= (pos[i] - pos[iSelf]);
        }
    }

	return seperation_force * rule2Scale;
}

__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
	glm::vec3 perceived_velocity(0.0f);
    int num_neighbors = 0;
    for(int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule3Distance) {
            perceived_velocity += vel[i];
			num_neighbors++;
        }
	}
    if (num_neighbors == 0) {
        return glm::vec3(0.0f);
    }
	perceived_velocity /= num_neighbors;
	return perceived_velocity * rule3Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 v1 = rule1(N, iSelf, pos);
	glm::vec3 v2 = rule2(N, iSelf, pos);
	glm::vec3 v3 = rule3(N, iSelf, pos, vel);

  return vel[iSelf] + v1 + v2 + v3;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if(index >= N){return;}

	glm::vec3 speedVec = computeVelocityChange(N, index, pos, vel1);
    //don't need to mult by maxSpeed but for style sake kept it
    if(glm::length(speedVec) > maxSpeed) {
       speedVec = glm::normalize(speedVec) * maxSpeed;
    }
    
    vel2[index] = speedVec;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) {
        return;
	}

    glm::vec3 curr_boid_pos = pos[index];

    int cellX = (curr_boid_pos.x - gridMin.x) * inverseCellWidth;
    int cellY =  (curr_boid_pos.y - gridMin.y) * inverseCellWidth;
    int cellZ = (curr_boid_pos.z - gridMin.z) * inverseCellWidth;

	//tranform and store boid position into grid index
    gridIndices[index] = gridIndex3Dto1D(cellX, cellY, cellZ, gridResolution);

    //store boid index (essentially boid id)
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  
	int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= N) {
        return;
    }

	int curr_cell = particleGridIndices[index];
    
	//if this is the fisrt boid or the previous boid is in a different cell then:
	// mark this boid as the start of the current cell
    if(index == 0 || curr_cell != particleGridIndices[index - 1]) {
        gridCellStartIndices[curr_cell] = index;
	}

	// if this is the last boid or the next boid is in a different cell then: 
    // mark the end of the current cell as the next index
    if (index == N - 1 ||
        particleGridIndices[index] != particleGridIndices[index + 1]) {

        gridCellEndIndices[curr_cell] = index + 1;
    }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) {
        return;
    }

    glm::vec3 curr_boid_pos = pos[index];

    int cellX = (curr_boid_pos.x - gridMin.x) * inverseCellWidth;
    int cellY = (curr_boid_pos.y - gridMin.y) * inverseCellWidth;
    int cellZ = (curr_boid_pos.z - gridMin.z) * inverseCellWidth;

    glm::vec3 perceivedCenter(0.0f);
    glm::vec3 separationForce(0.0f);
    glm::vec3 perceivedVelocity(0.0f);

    //rules may have different number of neighbors
    int R1neighbors = 0;
    int R3neighbors = 0;


    //iterate the 27 neiighboring cells
    for (int x = cellX - 1; x <= cellX + 1; x++) {
        for (int y = cellY - 1; y <= cellY + 1; y++) {
            for (int z = cellZ - 1; z <= cellZ + 1; z++) {

                if (x < 0 || x >= gridResolution ||
                    y < 0 || y >= gridResolution ||
                    z < 0 || z >= gridResolution) {
                    continue;
                }

                int cell =
                    gridIndex3Dto1D(x, y, z, gridResolution);

                int start = gridCellStartIndices[cell];
                int end = gridCellEndIndices[cell];

                if (start == -1) {
                    continue;
                }

                for (int j = start; j < end; j++) {

                    int neighborIndex = particleArrayIndices[j];

                    if (neighborIndex == index) {
                        continue;
                    }

                    glm::vec3 neighborPos = pos[neighborIndex];

                    float distance = glm::distance(curr_boid_pos, neighborPos);

                    if (distance < rule1Distance) {
                        perceivedCenter += neighborPos;
                        R1neighbors++;
                    }

                    if (distance < rule2Distance) {
                        separationForce -= neighborPos - curr_boid_pos;
                    }

                    if (distance < rule3Distance) {
                        perceivedVelocity += vel1[neighborIndex];
                        R3neighbors++;
                    }
                }
            }
        }
    }

    glm::vec3 v1(0.0f);
    glm::vec3 v2(0.0f);
    glm::vec3 v3(0.0f);

    if (R1neighbors > 0) {
        perceivedCenter /= R1neighbors;
        v1 = (perceivedCenter - curr_boid_pos) * rule1Scale;
    }

    v2 = separationForce * rule2Scale;

    if (R3neighbors > 0) {
        perceivedVelocity /= R3neighbors;
        v3 = perceivedVelocity * rule3Scale;
    }

    glm::vec3 newVelocity =
        vel1[index] + v1 + v2 + v3;

    //clamp
    if (glm::length(newVelocity) > maxSpeed) {
        newVelocity = glm::normalize(newVelocity) * maxSpeed;
    }

    vel2[index] = newVelocity;

}



//helper kernel to pre-sort the pos and vel arrays based on the particleArrayIndices
__global__ void kernReorderData(
    int N,
    int* particleArrayIndices,
    glm::vec3* pos,
    glm::vec3* vel1,
    glm::vec3* vel2,
    glm::vec3* pos_sorted,
    glm::vec3* vel1_sorted,
    glm::vec3* vel2_sorted
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) {
        return;
    }

    int originalIndex = particleArrayIndices[index];

    pos_sorted[index] = pos[originalIndex];
    vel1_sorted[index] = vel1[originalIndex];
    vel2_sorted[index] = vel2[originalIndex];
}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) {
        return;
    }

    glm::vec3 curr_boid_pos = pos[index];

    int cellX = (curr_boid_pos.x - gridMin.x) * inverseCellWidth;
    int cellY = (curr_boid_pos.y - gridMin.y) * inverseCellWidth;
    int cellZ = (curr_boid_pos.z - gridMin.z) * inverseCellWidth;

    glm::vec3 perceivedCenter(0.0f);
    glm::vec3 separationForce(0.0f);
    glm::vec3 perceivedVelocity(0.0f);

    //rules may have different number of neighbors
    int R1neighbors = 0;
    int R3neighbors = 0;


    //iterate the 27 neiighboring cells
	//we reorder the loops to maximize sequential memory access
    for (int z = cellZ - 1; z <= cellZ + 1; z++) {
        for (int y = cellY - 1; y <= cellY + 1; y++) {
            for (int x = cellX - 1; x <= cellX + 1; x++) {
       
                if (x < 0 || x >= gridResolution ||
                    y < 0 || y >= gridResolution ||
                    z < 0 || z >= gridResolution) {
                    continue;
                }

                int cell =
                    gridIndex3Dto1D(x, y, z, gridResolution);

                int start = gridCellStartIndices[cell];
                int end = gridCellEndIndices[cell];

                if (start == -1) {
                    continue;
                }

                for (int j = start; j < end; j++) {
                    //remove middleman buffer and index directly 
                    if (j == index) {
                        continue;
                    }

                    glm::vec3 neighborPos = pos[j];

                    float distance = glm::distance(curr_boid_pos, neighborPos);

                    if (distance < rule1Distance) {
                        perceivedCenter += neighborPos;
                        R1neighbors++;
                    }

                    if (distance < rule2Distance) {
                        separationForce -= neighborPos - curr_boid_pos;
                    }

                    if (distance < rule3Distance) {
                        perceivedVelocity += vel1[j];
                        R3neighbors++;
                    }
                }
            }
        }
    }

    glm::vec3 v1(0.0f);
    glm::vec3 v2(0.0f);
    glm::vec3 v3(0.0f);

    if (R1neighbors > 0) {
        perceivedCenter /= R1neighbors;
        v1 = (perceivedCenter - curr_boid_pos) * rule1Scale;
    }

    v2 = separationForce * rule2Scale;

    if (R3neighbors > 0) {
        perceivedVelocity /= R3neighbors;
        v3 = perceivedVelocity * rule3Scale;
    }

    glm::vec3 newVelocity =
        vel1[index] + v1 + v2 + v3;

    //clamp
    if (glm::length(newVelocity) > maxSpeed) {
        newVelocity = glm::normalize(newVelocity) * maxSpeed;
    }

    vel2[index] = newVelocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    kernUpdateVelocityBruteForce << <(numObjects + blockSize - 1)/blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);

    std::swap(dev_vel1, dev_vel2);

	kernUpdatePos <<<(numObjects + blockSize - 1)/blockSize>>> (numObjects, dt, dev_pos, dev_vel1);
	
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
    
    dim3 fullBlocksPerGrid(
        (numObjects + blockSize - 1) / blockSize
    );

    dim3 gridBlocks(
        (gridCellCount + blockSize - 1) / blockSize
    );

	//set up by finding indices, sorting, reseting cell ranges, and finding cell ranges
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices
        );

 
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices
    );

    kernResetIntBuffer <<<gridBlocks, blockSize >>> (gridCellCount, dev_gridCellStartIndices, -1);

    kernResetIntBuffer << <gridBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices,-1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices
    );


	// Perform velocity updates with neighbor search
    kernUpdateVelNeighborSearchScattered << < fullBlocksPerGrid, blockSize >> > (
            numObjects,
            gridSideCount,
            gridMinimum,
            gridInverseCellWidth,
            gridCellWidth,
            dev_gridCellStartIndices,
            dev_gridCellEndIndices,
            dev_particleArrayIndices,
            dev_pos,
            dev_vel1,
            dev_vel2
    );

    //ping pong
    std::swap(dev_vel1, dev_vel2);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dt,
        dev_pos,
        dev_vel1
   );


}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

    dim3 fullBlocksPerGrid(
        (numObjects + blockSize - 1) / blockSize
    );

    dim3 gridBlocks(
        (gridCellCount + blockSize - 1) / blockSize
    );

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices
        );

    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices
    );

    kernResetIntBuffer << <gridBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);

    kernResetIntBuffer << <gridBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	//use helper kernel to reorder
    kernReorderData << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_vel2,
        dev_pos_sorted,
        dev_vel1_sorted,
        dev_vel2_sorted
     );

    //swap into sorted buffers
    std::swap(dev_pos, dev_pos_sorted);
    std::swap(dev_vel1, dev_vel1_sorted);
    std::swap(dev_vel2, dev_vel2_sorted);

	//run search with coherent data
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_pos,
        dev_vel1,
        dev_vel2
    );

	std::swap(dev_vel1, dev_vel2);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (
        numObjects,
        dt,
        dev_pos,
        dev_vel1
    );
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_pos_sorted);
  cudaFree(dev_vel1_sorted);
  cudaFree(dev_vel2_sorted);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
