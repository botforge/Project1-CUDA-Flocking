#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

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
#define scene_scale 300.0f

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

// 2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_posShuffle;
glm::vec3* dev_vel1Shuffle;
glm::vec3* dev_vel2Shuffle;

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

	// LOOK-1.2 - This is a typical CUDA kernel invocation.
	// Initializes position of all the individual Boids in GPU
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

  // 2.1 Additional Buffers
	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndicies failed!");

	cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndicies failed!");

	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndicies failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndicies failed!");

  // 2.3 Additional Buffers
	cudaMalloc((void**)&dev_posShuffle, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_posShuffle failed!");

	cudaMalloc((void**)&dev_vel1Shuffle, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1Shuffle failed!");
	
	cudaMalloc((void**)&dev_vel2Shuffle, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2Shuffle failed!");

// Wrap the key/value buffers around the thrust pointers
	dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

  // 2.2 Additional Buffers
	cudaDeviceSynchronize();
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

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	//Get pos & vel of boid @ iSelf
	glm::vec3 iPos = pos[iSelf];
	glm::vec3 iVel = vel[iSelf];

	//Initialize Neighbors & Velocities
	glm::vec3 perceived_center(0.0f);
	float neighbors_rule1 = 0.f, neighbors_rule3 = 0.f;
	glm::vec3 perceived_velocity(0.0f);
	glm::vec3 vel_rule1(0.0f), vel_rule2(0.0f), vel_rule3(0.0f);
	glm::vec3 c(0.0f);
	for (int i = 0; i < N; ++i) {
		glm::vec3 boid_iPos = pos[i];
		glm::vec3 boid_iVel = vel[i];
		if (i == iSelf) continue;
		float boid_distance = glm::distance(iPos, boid_iPos);
		// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
		if (boid_distance < rule1Distance) {
			perceived_center += boid_iPos;
			neighbors_rule1 += 1;
		}
		// Rule 2: boids try to stay a distance d away from each other
		if (boid_distance < rule2Distance) {
			c -= boid_iPos - iPos;
		}
		// Rule 3: boids try to match the speed of surrounding boids
		if (boid_distance < rule3Distance) {
			perceived_velocity += boid_iVel;
			neighbors_rule3 += 1;
		}
	}
	//Updates for each Rule 
	vel_rule1 = (neighbors_rule1 > 0) ? ((perceived_center / neighbors_rule1) - iPos) * rule1Scale : glm::vec3(0.0f);
	vel_rule2 = c * rule2Scale;
	vel_rule3 = (neighbors_rule3 > 0) ? (perceived_velocity / neighbors_rule3) * rule3Scale : glm::vec3(0.0f);
	return vel_rule1 + vel_rule2 + vel_rule3;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) {
	glm::vec3 newVel(0.0f);
	int iSelf = threadIdx.x + (blockIdx.x * blockDim.x);
	if (iSelf >= N) {
		return;
	}
	// Compute a new velocity based on pos and vel1
	newVel = vel1[iSelf] + computeVelocityChange(N, iSelf, pos, vel1);
	// Clamp the speed & record new velocity into vel2
	vel2[iSelf] = glm::length(newVel) > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;
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
	// 2.1
	//   boid data in pos and vel1/vel2
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) {
		return;
	}
	glm::vec3 iPos = pos[index];

	//Label each boid with the index of its grid cell.
	glm::ivec3 cellPos = glm::floor((iPos - gridMin) * inverseCellWidth);
	gridIndices[index] = gridIndex3Dto1D(cellPos.x, cellPos.y, cellPos.z, gridResolution);
	//Set up a parallel array of integer indices as pointers to the actual
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
	// 2.1
	int boid_index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (boid_index >= N) {
		return;
	}
	// Identify the start point of each cell in the gridIndices array.
	int currCellIndex = particleGridIndices[boid_index];
	int prevCellIndex = (boid_index > 0) ? particleGridIndices[boid_index-1] : -1;
	int nextCellIndex = (boid_index < N - 1) ? particleGridIndices[boid_index + 1] : -1;
	bool isNewStart = currCellIndex != prevCellIndex;
	bool isNewEnd = currCellIndex != nextCellIndex || nextCellIndex == -1;
	if (isNewStart) {
		gridCellStartIndices[currCellIndex] = boid_index;
	}
	if (isNewEnd) {
		gridCellEndIndices[currCellIndex] = boid_index;
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  //Clamp the speed change before putting the new speed in vel2
	int boid_index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (boid_index >= N) {
		return;
	}
	//Identify the grid cell that this particle is in
	glm::vec3 iPos = pos[boid_index];
	glm::vec3 iPos_cellSpace = (iPos - gridMin) * inverseCellWidth;
	glm::vec3 offset_cellSpace = glm::fract(iPos_cellSpace);
	glm::ivec3 cellPos_3D = glm::floor(iPos_cellSpace);

	//Identify which cells may contain neighbors
	//We do this by considering which octant the boid is in & using that to decide neighbor directions
	//If x,y or z are > 0 this means they were > 0.5
	glm::vec3 neighbor_dirs = offset_cellSpace - glm::vec3(0.5); 
	glm::ivec3 cellMin_3D(cellPos_3D);
	glm::ivec3 cellMax_3D(cellPos_3D);
	//Min Bounds in CellSpace
	cellMin_3D.x = (neighbor_dirs.x < 0 && cellPos_3D.x >= 0) ? cellPos_3D.x -1 : cellPos_3D.x;
	cellMin_3D.y = (neighbor_dirs.y < 0 && cellPos_3D.y >= 0) ? cellPos_3D.y -1 : cellPos_3D.y;
	cellMin_3D.z = (neighbor_dirs.z < 0 && cellPos_3D.z >= 0) ? cellPos_3D.z -1 : cellPos_3D.z;
	//Max Bounds in CellSpace
	cellMax_3D.x = (neighbor_dirs.x >= 0 && cellPos_3D.x < gridResolution) ? cellPos_3D.x +1 : cellPos_3D.x;
	cellMax_3D.y = (neighbor_dirs.y >= 0 && cellPos_3D.y < gridResolution) ? cellPos_3D.y +1 : cellPos_3D.y;
	cellMax_3D.z = (neighbor_dirs.z >= 0 && cellPos_3D.z < gridResolution) ? cellPos_3D.z +1 : cellPos_3D.z;

	//Initialize Neighbors & Velocities
	glm::vec3 perceived_center(0.0f);
	float neighbors_rule1 = 0.f, neighbors_rule3 = 0.f;
	glm::vec3 perceived_velocity(0.0f);
	glm::vec3 vel_rule1(0.0f), vel_rule2(0.0f), vel_rule3(0.0f);
	glm::vec3 c(0.0f);

	//Loop over gridindicies
	for (int z = cellMin_3D.z; z <= cellMax_3D.z; ++z) {
		for (int y = cellMin_3D.y; y <= cellMax_3D.y; ++y) {
			for (int x = cellMin_3D.x; x <= cellMax_3D.x; ++x) {
				int currCell_1D = gridIndex3Dto1D(x, y, z, gridResolution);
				int cell_startidx = gridCellStartIndices[currCell_1D];
				int cell_endidx = gridCellEndIndices[currCell_1D];

				//For each cell, read the start/end indices in the boid pointer array.
				for (int b_idx = cell_startidx; b_idx < cell_endidx; b_idx++) {
					int b = particleArrayIndices[b_idx];
					if (b == boid_index) { continue; }
					glm::vec3 boid_iPos = pos[b];
					glm::vec3 boid_iVel = vel1[b];
					float boid_distance = glm::distance(iPos, boid_iPos);
					//Access each boid in the cell and compute velocity change from the boids rules, if this boid is within the neighborhood distance.
					// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
					if (boid_distance < rule1Distance) {
						perceived_center += boid_iPos;
						neighbors_rule1 += 1;
					}
					// Rule 2: boids try to stay a distance d away from each other
					if (boid_distance < rule2Distance) {
						c -= boid_iPos - iPos;
					}
					// Rule 3: boids try to match the speed of surrounding boids
					if (boid_distance < rule3Distance) {
						perceived_velocity += boid_iVel;
						neighbors_rule3 += 1;
					}
				}
			}
		}
	}
	//Updates for each Rule 
	vel_rule1 = (neighbors_rule1 > 0) ? ((perceived_center / neighbors_rule1) - iPos) * rule1Scale : glm::vec3(0.0f);
	vel_rule2 = c * rule2Scale;
	vel_rule3 = (neighbors_rule3 > 0) ? (perceived_velocity / neighbors_rule3) * rule3Scale : glm::vec3(0.0f);
	glm::vec3 newVel = vel_rule1 + vel_rule2 + vel_rule3;
	// Compute a new velocity based on pos and vel1
	newVel += vel1[boid_index];
	// Clamp the speed & record new velocity into vel2
	vel2[boid_index] = glm::length(newVel) > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;
 }

__device__ void shuffleArray(int curr_idx, int* indicies, glm::vec3* orig, glm::vec3* shuffle) {
	shuffle[curr_idx] = orig[indicies[curr_idx]];
}
__global__ void kernShuffleData(int N, int *particleArrayIndicies, glm::vec3* pos, glm::vec3* posShuffle, glm::vec3* vel1, glm::vec3* vel1Shuffle, glm::vec3* vel2, glm::vec3* vel2Shuffle) {
	int boid_index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (boid_index >= N) {
		return;
	}
	shuffleArray(boid_index, particleArrayIndicies, pos, posShuffle);
	shuffleArray(boid_index, particleArrayIndicies, vel1, vel1Shuffle);
	shuffleArray(boid_index, particleArrayIndicies, vel2, posShuffle);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  //Clamp the speed change before putting the new speed in vel2
	int boid_index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (boid_index >= N) {
		return;
	}
	//Identify the grid cell that this particle is in
	glm::vec3 iPos = pos[boid_index];
	glm::vec3 iPos_cellSpace = (iPos - gridMin) * inverseCellWidth;
	glm::vec3 offset_cellSpace = glm::fract(iPos_cellSpace);
	glm::ivec3 cellPos_3D = glm::floor(iPos_cellSpace);

	//Identify which cells may contain neighbors
	//We do this by considering which octant the boid is in & using that to decide neighbor directions
	//If x,y or z are > 0 this means they were > 0.5
	glm::vec3 neighbor_dirs = offset_cellSpace - glm::vec3(0.5); 
	glm::ivec3 cellMin_3D(cellPos_3D);
	glm::ivec3 cellMax_3D(cellPos_3D);
	//Min Bounds in CellSpace
	cellMin_3D.x = (neighbor_dirs.x < 0 && cellPos_3D.x >= 0) ? cellPos_3D.x -1 : cellPos_3D.x;
	cellMin_3D.y = (neighbor_dirs.y < 0 && cellPos_3D.y >= 0) ? cellPos_3D.y -1 : cellPos_3D.y;
	cellMin_3D.z = (neighbor_dirs.z < 0 && cellPos_3D.z >= 0) ? cellPos_3D.z -1 : cellPos_3D.z;
	//Max Bounds in CellSpace
	cellMax_3D.x = (neighbor_dirs.x >= 0 && cellPos_3D.x < gridResolution) ? cellPos_3D.x +1 : cellPos_3D.x;
	cellMax_3D.y = (neighbor_dirs.y >= 0 && cellPos_3D.y < gridResolution) ? cellPos_3D.y +1 : cellPos_3D.y;
	cellMax_3D.z = (neighbor_dirs.z >= 0 && cellPos_3D.z < gridResolution) ? cellPos_3D.z +1 : cellPos_3D.z;

	//Initialize Neighbors & Velocities
	glm::vec3 perceived_center(0.0f);
	float neighbors_rule1 = 0.f, neighbors_rule3 = 0.f;
	glm::vec3 perceived_velocity(0.0f);
	glm::vec3 vel_rule1(0.0f), vel_rule2(0.0f), vel_rule3(0.0f);
	glm::vec3 c(0.0f);

	//Loop over gridindicies
	for (int z = cellMin_3D.z; z <= cellMax_3D.z; ++z) {
		for (int y = cellMin_3D.y; y <= cellMax_3D.y; ++y) {
			for (int x = cellMin_3D.x; x <= cellMax_3D.x; ++x) {
				int currCell_1D = gridIndex3Dto1D(x, y, z, gridResolution);
				int cell_startidx = gridCellStartIndices[currCell_1D];
				int cell_endidx = gridCellEndIndices[currCell_1D];

				//For each cell, read the start/end indices in the boid pointer array.
				for (int b_idx = cell_startidx; b_idx < cell_endidx; b_idx++) {
					int b = particleArrayIndices[b_idx];
					if (b == boid_index) { continue; }
					glm::vec3 boid_iPos = pos[b];
					glm::vec3 boid_iVel = vel1[b];
					float boid_distance = glm::distance(iPos, boid_iPos);
					//Access each boid in the cell and compute velocity change from the boids rules, if this boid is within the neighborhood distance.
					// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
					if (boid_distance < rule1Distance) {
						perceived_center += boid_iPos;
						neighbors_rule1 += 1;
					}
					// Rule 2: boids try to stay a distance d away from each other
					if (boid_distance < rule2Distance) {
						c -= boid_iPos - iPos;
					}
					// Rule 3: boids try to match the speed of surrounding boids
					if (boid_distance < rule3Distance) {
						perceived_velocity += boid_iVel;
						neighbors_rule3 += 1;
					}
				}
			}
		}
	}
	//Updates for each Rule 
	vel_rule1 = (neighbors_rule1 > 0) ? ((perceived_center / neighbors_rule1) - iPos) * rule1Scale : glm::vec3(0.0f);
	vel_rule2 = c * rule2Scale;
	vel_rule3 = (neighbors_rule3 > 0) ? (perceived_velocity / neighbors_rule3) * rule3Scale : glm::vec3(0.0f);
	glm::vec3 newVel = vel_rule1 + vel_rule2 + vel_rule3;
	// Compute a new velocity based on pos and vel1
	newVel += vel1[boid_index];
	// Clamp the speed & record new velocity into vel2
	vel2[boid_index] = glm::length(newVel) > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	//Update Velocity, then Pos by calling appropriate Kernels
	kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos,
		dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	//Ping Pong the velocity buffers
	cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}


void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndicies failed!");

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer1 failed!");
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer2 failed!");
	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	//Perform Velocity Updates
	kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	//Ping Pong the velocity buffers
	cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
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

  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndicies failed!");

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer1 failed!");
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer2 failed!");

	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	kernShuffleData<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices, dev_pos, dev_posShuffle, dev_vel1, dev_vel1Shuffle, dev_vel2, dev_vel2Shuffle);
	checkCUDAErrorWithLine("kernShuffleData failed!");
	cudaDeviceSynchronize();
	cudaMemcpy(dev_pos, dev_posShuffle, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_vel1, dev_vel1Shuffle, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_vel2, dev_vel2Shuffle, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	//Perform Velocity Updates
	kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	//Ping Pong the velocity buffers
	cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  //2.1 Allocated Buffers
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices); 

  //2.3 Allocated Buffers
  cudaFree(dev_posShuffle);
  cudaFree(dev_vel1Shuffle);
  cudaFree(dev_vel2Shuffle);
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