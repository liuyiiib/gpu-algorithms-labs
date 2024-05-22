#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define WARP_SIZE 32
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to global queue
  unsigned int nCurrLevel = *numCurrLevelNodes;
  int tId = blockDim.x * blockIdx.x + threadIdx.x;
  if (tId == 0) {
    *numNextLevelNodes = 0;
  }
  __syncthreads();
  if (tId < nCurrLevel) {
    unsigned int node = currLevelNodes[tId];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      unsigned int visit = nodeVisited[neighborID];
      if (!visit) {
        if (atomicCAS(&nodeVisited[neighborID], visit, 1) == visit) {
          int tail = atomicAdd(numNextLevelNodes, 1);  // get the current tail
          nextLevelNodes[tail] = neighborID;
        }
      }
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue (size should be BQ_CAPACITY)
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int blocktail, globaltail;
  unsigned int nCurrLevel = *numCurrLevelNodes;
  int tId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadIdx.x == 0) {
    blocktail = 0;
  }
  if (tId == 0) {
    *numNextLevelNodes = 0;
  }
  __syncthreads();
  if (tId < nCurrLevel) {
    unsigned int node = currLevelNodes[tId];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      unsigned int visit = nodeVisited[neighborID];
      if (!visit) {
        if (atomicCAS(&nodeVisited[neighborID], visit, 1) == visit) {
          int tail = atomicAdd(&blocktail, 1);  // get the current tail
          if(tail < BQ_CAPACITY){
            blockQueue[tail] = neighborID;
          }
          else{
            tail = BQ_CAPACITY;
            nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighborID;
          }
        }
      }
    }
  }
  __syncthreads();
  if(threadIdx.x == 0)
    globaltail = atomicAdd(numNextLevelNodes, blocktail);
  __syncthreads();
  // int perthread = ceil(blocktail / blockDim.x);
  for (int i = threadIdx.x; i < blocktail; i += blockDim.x) {
    nextLevelNodes[globaltail + i] = blockQueue[i];
  }
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to block queue
  // If full, add neighbor to global queue

  // Allocate space for block queue to go into global queue

  // Store block queue in global queue
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  __shared__ unsigned int warpQueue[WQ_CAPACITY][NUM_WARP_QUEUES];
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int warpQueueTail[NUM_WARP_QUEUES];
  __shared__ unsigned int warpQueuePrefix[NUM_WARP_QUEUES];
  __shared__ unsigned int blockQueueTail, blockToGlobalQueueTail;
  int tId = blockDim.x * blockIdx.x + threadIdx.x;
  if(threadIdx.x / NUM_WARP_QUEUES == 0){
    warpQueueTail[threadIdx.x] = 0;
    if(threadIdx.x == 0){
      blockQueueTail = 0;
      if(tId == 0)
        *numNextLevelNodes = 0;
    }
  }
  __syncthreads();
  unsigned int nCurrLevel = *numCurrLevelNodes;
  unsigned int warpId = threadIdx.x % NUM_WARP_QUEUES;

  if (tId < nCurrLevel) {
    unsigned int node = currLevelNodes[tId];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      if (!atomicExch(&nodeVisited[neighborID], 1)) {
        int tail = atomicAdd(&warpQueueTail[warpId], 1);  // get the current tail
        if(tail < WQ_CAPACITY){
          warpQueue[tail][warpId] = neighborID;
        }
        else{
          warpQueueTail[warpId] = WQ_CAPACITY;
          int btail = atomicAdd(&(blockQueueTail), 1);
          if (btail < BQ_CAPACITY) {
            blockQueue[btail] = neighborID;
          } else {
            blockQueueTail = BQ_CAPACITY;
            nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighborID;
          }
        }
      }
    }
  }

  __syncthreads();
  if(threadIdx.x == 0){
    warpQueuePrefix[0] = blockQueueTail;
    for(unsigned int i = 1;i < NUM_WARP_QUEUES;i++)
      warpQueuePrefix[i] = warpQueuePrefix[i - 1] + warpQueueTail[i - 1];
  }

  __syncthreads();
  if(threadIdx.x == 0){
    if(warpQueuePrefix[NUM_WARP_QUEUES - 1] + warpQueueTail[NUM_WARP_QUEUES - 1] < BQ_CAPACITY)
      blockQueueTail = warpQueuePrefix[NUM_WARP_QUEUES - 1] + warpQueueTail[NUM_WARP_QUEUES - 1];
    else
      blockQueueTail = BQ_CAPACITY; 
    blockToGlobalQueueTail = atomicAdd(numNextLevelNodes, blockQueueTail);
  }
  const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  if (warpIdx < NUM_WARP_QUEUES) {
    for(unsigned int i = threadIdx.x % WARP_SIZE; i < warpQueueTail[warpIdx];i += WARP_SIZE){
      const unsigned int warpToBlockQueueIdx = warpQueuePrefix[warpIdx] + i;
      if(warpToBlockQueueIdx < BQ_CAPACITY)
        blockQueue[warpToBlockQueueIdx] = warpQueue[i][warpIdx];
      else
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = warpQueue[i][warpIdx];
    }
  }
  __syncthreads();
  for(unsigned int i = threadIdx.x;i < blockQueueTail;i += blockDim.x)
    nextLevelNodes[blockToGlobalQueueTail + i] = blockQueue[i];


  
  // This version uses NUM_WARP_QUEUES warp queues of capacity 
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.  

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.

  // Initialize shared memory queues (warp and block)

  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to the queue
  // If full, add neighbor to block queue
  // If full, add neighbor to global queue

  // Allocate space for warp queue to go into block queue

  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue

  // Saturate block queue counter (too large if warp queues overflowed)
  // Allocate space for block queue to go into global queue

  // Store block queue in global queue
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel <<<numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel <<<numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel <<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
