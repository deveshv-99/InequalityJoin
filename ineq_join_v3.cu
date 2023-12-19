#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/async/copy.h>
#include <thrust/async/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_scan.h>
#include <numeric>
#include <chrono>

typedef long long int lld;

#define R_SIZE 1000000
#define S_SIZE 1000000


int r_key[R_SIZE],s_key[S_SIZE];
int r_val[R_SIZE],s_val[S_SIZE];

// int r_key[]={0,1,2,3,4,5,6,7,8,9};
// int r_val[]={1,-1,3,5,-3,-5,7,9,-7,-9};

// int s_key[]={0,1,2,3,4,5,6};
// int s_val[]={0,2,-2,4,-4,6,8};

lld offset[R_SIZE+1];

// Function to read data from table R
void readTableR(){
    std::ifstream file("R_table.csv");
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }


    int i=0;
    std::string a;
    while(file >> a){
            int j=0;
            while(a[j] != ','){
                j++; 
            }
            std::string str1 = a.substr(0,j);
            std::string str2 = a.substr(j+1,a.size()-j-1);
            r_key[i] = std::stoi(str1);
            r_val[i] = std::stoi(str2);
           
        i++;
    }
    file.close();
}

// Function to read data from table S
void readTableS(){
    std::ifstream file("S_table.csv");
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }


    int i=0;
    std::string a;
    while(file >> a){
            int j=0;
            while(a[j] != ','){
                j++; 
            }
            std::string str1 = a.substr(0,j);
            std::string str2 = a.substr(j+1,a.size()-j-1);
            s_key[i] = std::stoi(str1);
            s_val[i] = std::stoi(str2);
        i++;
    }
    file.close();
}



//------------------------------INCOMPLETE-----------------------------------------------------
__global__ void computeOffsets(int* r_val, int* s_val, int* offset, int r_size, int s_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < r_size) {
        int low = 0, high = s_size - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (s_val[mid] <= r_val[i])
                low = mid + 1;
            else
                high = mid - 1;
        }
        offset[i] = low; // 'low' is the index where r_val[i] would be inserted in s_val
    }
}


__global__ void buildTable(int *d_rkey, int *d_skey, lld *d_prefix, int *d_output,int r_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= r_size)
        return;
    
    lld startIndex = d_prefix[idx];
    lld endIndex = d_prefix[idx+1];

    // if(idx==1){
    //     for(int i=0;i<=r_size;i++){
    //         printf("%lld ",d_prefix[i]);
    //     }
    //     printf("\t Thread ID is: %d\n",idx);
    //     printf("\t Start Index is: %lld\n",startIndex);
    //     printf("\t End Index is: %lld\n",endIndex);
    //     printf("\t rkey is: %d\n",d_rkey[idx]);
    // }
    
    if(startIndex == endIndex)
        return;
    int j=0;
    
    for(int i=startIndex;i<endIndex;i++){
        d_output[2*i] = d_rkey[idx];
        d_output[2*i+1] = d_skey[j++];
    }
    
}

//------------------------------INCOMPLETE-----------------------------------------------------

int main() {

    // Read data from CSV files
    readTableR();
    readTableS();

    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();
    
    ////// Yaha GPU memory me r_val aur r_key data transfer karna hai
    thrust::sort_by_key(thrust::device,r_val, r_val + R_SIZE, r_key, thrust::greater<int>());
    thrust::sort_by_key(s_val, s_val + S_SIZE, s_key, thrust::greater<int>());
    

    

    int *d_rkey, *d_skey, *d_output;
    lld *d_prefix;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_rval, R_SIZE * sizeof(int));
    cudaMalloc((void **)&d_sval, S_SIZE * sizeof(int));
    cudaMalloc((void **)&d_off, (R_SIZE+1) * sizeof(int));


    // Copy data from host to device
    cudaMemcpy(d_rval, r_val, R_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sval, s_val, S_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //Launch the kernel
    int threadsPerBlock = 1024;
    int blocks = (R_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    computeOffsets<<<blocks, threadsPerBlock>>>(d_rval, d_sval, d_off, R_SIZE, S_SIZE);


    outputSize+=offset[R_SIZE-1];
    lld outputSize = offset[R_SIZE-1] ;
    // for (int i = 0; i < R_SIZE; ++i) {
    //     std::cout << offset[i] << " ";
    // }
    // std::cout << std::endl;
    
    thrust::device_vector<lld> d_offset(offset, offset + R_SIZE);
    //thrust::inclusive_scan(d_offset.begin(), d_offset.end(), d_offset.begin());

    ;
    //std::cout<<firstIndex<<std::endl;

    thrust::exclusive_scan(d_offset.begin(), d_offset.end(), d_offset.begin());
    

    thrust::copy(d_offset.begin(), d_offset.end(), offset);
    
    
    outputSize+=offset[R_SIZE-1]; 
    offset[R_SIZE] = outputSize;


    if(outputSize> 1<<30)
    {
        std::cout<<"Number of tuples in the Join relation is: "<<outputSize<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        printf("Time measured: %.3f ms.\n", elapsed.count() * 1e-6);
        return 0;
    }



    //------------------------KERNEL STUFF------------------------------------

    
    
    int *d_rkey, *d_skey, *d_output;
    lld *d_prefix;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_rkey, R_SIZE * sizeof(int));
    cudaMalloc((void **)&d_skey, S_SIZE * sizeof(int));
    cudaMalloc((void **)&d_prefix, (R_SIZE+1) * sizeof(lld));
    cudaMalloc((void **)&d_output, 2*outputSize * sizeof(int));


    // Copy data from host to device
    cudaMemcpy(d_rkey, r_key, R_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_skey, s_key, S_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, offset, (R_SIZE+1) * sizeof(lld), cudaMemcpyHostToDevice);


    //Launch the kernel
    int threadsPerBlock = 1024;
    int blocks = (R_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    buildTable<<<blocks, threadsPerBlock>>>(d_rkey, d_skey, d_prefix, d_output,R_SIZE);
    //buildTable<<<1, 2>>>(d_rkey, d_skey, d_prefix, d_output,R_SIZE);


    // Copy results back to host
    int *output = (int *)malloc(2*outputSize* sizeof(int));
    cudaMemcpy(output, d_output, 2*outputSize* sizeof(int), cudaMemcpyDeviceToHost);

    
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %s after launching myKernel!\n", cudaGetErrorString(cudaStatus));
    }
    
    // Cleanup
    cudaFree(d_rkey);
    cudaFree(d_skey);
    cudaFree(d_prefix);
    cudaFree(d_output);


    // std::ofstream myfile;
    // myfile.open ("result.csv");
    // for (int i = 0; i < 2*outputSize; i+=2) {
    //     myfile << output[i] << "," << output[i+1] << "\n";
    // }
    // myfile.close();

    std::cout<<"Number of tuples in the Join relation is: "<<outputSize<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    free(output);
    
    // Stop measuring time and calculate the elapsed time
    
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
 
    printf("Time measured: %.3f ms.\n", elapsed.count() * 1e-6);

    return 0;
}
