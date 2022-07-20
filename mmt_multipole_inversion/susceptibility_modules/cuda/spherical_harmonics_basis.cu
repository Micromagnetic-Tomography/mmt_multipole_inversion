#include "spherical_harmonics_basis.cuh"
#include <cmath>
#include <math.h>
#include <stdio.h>


// Define magnetic constant in GPU
// __device__ double Cm = 1e-7;

// The implementation here uses a grid-stride loop:
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void SHB_pop_matrix(double * Q, double * r_sources, double * r_sensors,
                               unsigned long long Nsources,
                               unsigned long long Nsensors,
                               int multipole_order, int n_multipoles,
                               int verbose) {


    // The thread's unique number
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // int threadsInBlock = blockDim.x;

    for (unsigned long long n = global_idx; n < (Nsources * Nsensors); n += stride) {

        unsigned long long i_sensor = n / Nsources;
        unsigned long long i_source = n % Nsources;
        // printf("n = %ld isens = %ld isource = %ld\n", n, i_sensor, i_source);

        double x = r_sensors[3 * i_sensor    ] - r_sources[3 * i_source    ];
        double y = r_sensors[3 * i_sensor + 1] - r_sources[3 * i_source + 1];
        double z = r_sensors[3 * i_sensor + 2] - r_sources[3 * i_source + 2];
        double r2 = x * x + y * y + z * z;
        double r = sqrt(r2);

        // Multipole field susceptibilities; we will re-use this matrix using
        // the largest number of multipoles
        // double * p = (double *) malloc(sizeof(double) * (2 * multipole_order + 1));
        double * p = new double[2 * multipole_order + 1];
        int k;
        double f;

        // DIPOLE
        if (multipole_order > 0) {
            f = 1e-7 / (r2 * r2 * r);
            p[2] = (3 * z * z - r2);
            p[1] = (3 * y * z);
            p[0] = (3 * x * z);
            // Assign the 3 dipole entries in the 1st 3 entries of the Q matrix
            for (k = 0; k < 3; ++k) Q[n_multipoles * n + k] = f * p[k];

            // for (k = 0; k < 3; ++k) printf("%ld isn = %ld isrc = %ld k = %d Q = %.5e p = %.5e f = %.5e  r2 = %.5e x = %.5e y = %.5e\n",
            //                                n, i_sensor, i_source, k, Q[n_multipoles * n + k], p[k], f, r2, x, y);
        }

        // QUADRUPOLE
        if (multipole_order > 1) {
            double z2 = z * z;
            // Quad Field from the Cart version of Quad field SHs, by Stone et al
            f = 1e-7 / (r2 * r2 * r2 * r);
            p[0] = sqrt(3.0 / 2.0) * z * (-3 * r2 + 5 * z2);
            p[1] = -sqrt(2.0) * x * (r2 - 5 * z2);
            p[2] = -sqrt(2.0) * y * (r2 - 5 * z2);
            p[3] = (5.0 / sqrt(2.0)) * (x * x - y * y) * z;
            p[4] = 5.0 * sqrt(2.0) * x * y * z;

            for (k = 0; k < 5; ++k) Q[n_multipoles * n + k + 3] = f * p[k];
        }

        // OCTUPOLE
        if (multipole_order > 2) {
            // Oct Field from the Cartesian version of Octupole field SHs, by Stone et al
            double r4 = r2 * r2;
            double x2 = x * x;
            double y2 = y * y;
            double z2 = z * z;
            f = 1e-7 / (r4 * r4 * r);
            p[0] = (3 * r4 - 30 * r2 * z2 + 35 * (z2 * z2)) / sqrt(10.0);
            p[1] = sqrt(15.0) * x * z * (-3 * r2 + 7 * z2) / 2;
            p[2] = sqrt(15.0) * y * z * (-3 * r2 + 7 * z2) / 2;
            p[3] = -sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2);
            p[4] = -sqrt(6.0) * x * y * (r2 - 7 * z2);
            p[5] = 7 * x * (x2 - 3 * y2) * z / 2;
            p[6] = -7 * y * (-3 * x2 + y2) * z / 2;

            for (k = 0; k < 7; ++k) Q[n_multipoles * n + k + 8] = f * p[k];
        }

        // free(p);
        delete [] p;
    } // end for loop
}

/* Calculation of the magnetic susceptibility matrix Q using CUDA. This is
   done using a 1D array. The Q array has
   `n_mp * Nsources * Nsensors` elements with `n_mp` as the
   number of multipole tensor components: `2 * n_mp + 1`. The GPU loops over
   `Nsources * Nsensors` entries, where each GPU thread calculates all the
   multipole tensor comps.

   Parameters
   ----------
   r_sources
       N * 3 array with the positions of the magnetic sources
   r_sensors
       M * 3 array with the positions of the sensors (scan surface)
   Q
       Array to store the field susceptibilities
   Nsources, Nsensors
       Number of magnetic point sources and scan sensors
   multipole_order
       1 -> dipole, 2 -> quadrupole , ...
 */
void SHB_populate_matrix_cuda(double * r_sources,
                              double * r_sensors,
                              double * Q,
                              unsigned long long Nsources,
                              unsigned long long Nsensors,
                              int multipole_order,
                              int verbose
                              ) {

    // Total number of multipole values
    int n_multipoles = multipole_order * (multipole_order + 2);

    unsigned long long Qsize = n_multipoles * Nsources * Nsensors;
    // Each thread will compute `n_multipoles` elements
    unsigned long long Nsrc_x_Nsns = Nsources * Nsensors;

    size_t Q_bytes = sizeof(double) * Qsize;
    // Manual mem allocation: G in GPU and cuboids_dev in GPU
    double *Q_dev;
    // CUDA_ASSERT(cudaMalloc((void**)&Q_dev, Q_bytes));
    // (allocate in GPU if enough memory, see below)
    // cudaMalloc((void**)&Q_dev, Q_bytes);

    double *r_sources_dev;
    cudaMalloc((void**)&r_sources_dev, sizeof(double) * 3 * Nsources);
    // Copy cuboids array from the host to the GPU
    cudaMemcpy(r_sources_dev, r_sources, sizeof(double) * 3 * Nsources, cudaMemcpyHostToDevice);

    double *r_sensors_dev;
    cudaMalloc((void**)&r_sensors_dev, sizeof(double) * 3 * Nsensors);
    // Copy cuboids array from the host to the GPU
    cudaMemcpy(r_sensors_dev, r_sensors, sizeof(double) * 3 * Nsensors, cudaMemcpyHostToDevice);

    // // Launch kernel
    // // Quadro RTX 6000: 4608 CUDA Cores
    // // More refined matrix allocation of blocks if we use smaller n_threads, e.g. 8
    // // Use a N of threads multiple of 32 (multiple of warp size; see docs)
    // int blockSize = 256;
    // // Determine blocks and grid based on problem size:
    // // We will use the number of dipoles and sensors only, Q is larger in size
    // int gridSize = ceil(Nsrc_x_Nsns / (float) n_threads);
    // dim3 grid(gridSize, 1, 1);
    // dim3 block(blockSize, 1, 1);

    // Here we use LESS blocks so that threads can compute more efficiently,
    // taking advantage of the grid-stride loop and increasing occupancy
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch (full GPU)
    int gridSize;    // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       SHB_pop_matrix, 0, 0);
    // Round up according to calculation size
    gridSize = (Nsrc_x_Nsns + blockSize - 1) / blockSize;
    // If the processed array is larger than max thread capacity, use full GPU
    // rather than a large number of blocks
    gridSize = gridSize < minGridSize ? gridSize : minGridSize;

    // GPU properties:
    // int deviceId;
    // cudaGetDevice(&deviceId);
    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, deviceId);
    // printf("MultiProcessor count = %d\n", props.multiProcessorCount);
    // printf("Max threads per MP = %d\n", props.maxThreadsPerMultiProcessor);

    // Checking available memory in GPU:
    if(verbose == 1) {
        size_t free_byte;
        size_t total_byte;
        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
        double free_db = (double) free_byte / (1024. * 1024.);
        // Quadro RTX 6000: total mem should be 24220.3125 Mb
        double total_db = (double) total_byte / (1024. * 1024.);
        double used_db = total_db - free_db;
        double Q_size_mb = (double) Q_bytes / (1024. * 1024.);
        double r_sources_size_mb = (double) (3 * Nsources * sizeof(double)) / (1024. * 1024.);

        printf("------------ Nvidia GPU calculation info ------------\n");
        printf("GPU Memory      (MB): free  = %.4f\n", free_db);
        printf("                      used  = %.4f\n", used_db);
        printf("                      total = %.4f\n", total_db);
        printf("Size of Q       (MB): %.4f\n", Q_size_mb);
        printf("Size of r_sources   (MB): %.4f\n", r_sources_size_mb);
        printf("Grid size = %d\n", gridSize);
        printf("Block size (n threads / block) = %d\n", blockSize);
        // printf("Sensor Matrix dims (rows x cols) = %d x %d\n", (n_multipoles) * Nsrc_x_Nsns);
        printf("-----------------------------------------------------\n");
    }

    // Allocate G matrix
    cudaMalloc((void**)&Q_dev, Q_bytes);

    // Populate matrix in GPU:
    SHB_pop_matrix<<<gridSize, blockSize>>>(Q_dev, r_sources_dev, r_sensors_dev,
                                            Nsources, Nsensors,
                                            multipole_order, n_multipoles,
                                            verbose);
    cudaDeviceSynchronize();

    // Copy Q from the GPU to the host
    cudaMemcpy(Q, Q_dev, Q_bytes, cudaMemcpyDeviceToHost);

    // for (int k = 0; k < Nsensors; ++k) printf("%d %f\n", k, r_sensors[k]);
    // for (int k = 0; k < Nsrc_x_Nsns; ++k) printf("%f\n", Q[k]);

    cudaFree(Q_dev);
    cudaFree(r_sources_dev);
    cudaFree(r_sensors_dev);

} // main function
