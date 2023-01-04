#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

bool checkParameter(int width, int height, int numSeamRemoved)
{
    if(width <= 0 || height <= 0)
    {
        printf("You don't have any inputs!");
        return 1;
    }
    else if(numSeamRemoved < 0)
    {
        printf("Beyond the scope of this project!");
        return 1;
    }
    else if(numSeamRemoved == 0)
    {
        printf("You already have output!");
        return 1;
    }
    else if(numSeamRemoved == width)
    {
        printf("Here is your output......... Yes, nothing!");
        return 1;
    }
    return 0;
}

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) 
{
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) 
        {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

uint8_t getClosest(uint8_t *pixels, int r, int c, int width, int height)
{
    if (r < 0)
        r = 0;
    else if (r >= height)
        r = height - 1;

    if (c < 0)
        c = 0;
    else if (c >= width)
        c = width - 1;

    return pixels[r * width + c];
}

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

int computePixelPriority(uint8_t * grayPixels, int row, int col, int width, int height) 
{
    int x = 0, y = 0;
    for (int i = 0; i < 3; i++) 
        for (int j = 0; j < 3; j++) 
        {
            uint8_t closest = getClosest(grayPixels, row - 1 + i, col - 1 + j, width, height);
            x += closest * xSobel[i][j];
            y += closest * ySobel[i][j];
        }

    return abs(x) + abs(y);
}

void computeSeamScoreTable(int *priority, int *score, int *path, int width, int height) 
{
    for (int c = 0; c < width; c++)
        score[c] = priority[c];
    
    for (int r = 1; r < height; r++) 
        for (int c = 0; c < width; c++) 
        {
            int idx = r * width + c;
            int aboveIdx = (r - 1) * width + c;

            int min = score[aboveIdx];
            if (c > 0 && score[aboveIdx - 1] < min) 
            {
                min = score[aboveIdx - 1];
                path[idx] = -1;
            }
            
            if (c < width - 1 && score[aboveIdx + 1] < min) 
            {
                min = score[aboveIdx + 1];
                path[idx] = 1;
            }
            
            score[idx] = min + priority[idx];
        }
}

void seamCarvingByHost(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) 
{
    GpuTimer timer;
    timer.Start();

    // Allocate memory
    int *priority = (int *)malloc(width * height * sizeof(int));
    
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));

    uchar3 * tmpOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));

    memcpy(tmpOutPixels, inPixels, width * height * sizeof(uchar3));
    
    // Turn input image to grayscale
    convertRgb2Gray(inPixels, width, height, grayPixels);

    // Compute pixel priority
    for (int r = 0; r < height; r++) 
        for (int c = 0; c < width; c++) 
            priority[r * width + c] = computePixelPriority(grayPixels, r, c, width, height);

    while (width > targetWidth)
    {
        // Compute min seam table
        int *score = (int *)malloc(width * height * sizeof(int));       // Dynamic score table
        int *path = (int *)malloc(width * height * sizeof(int));        // Dynamic path table
        memset(path, 0, width * height * sizeof(int));                  // Set all path to 0

        computeSeamScoreTable(priority, score, path, width, height);    // Compute score and path

        uchar3 * newOutPixels = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));    // Allocate new picture after remove seam
        uint8_t *newGrayPixels= (uint8_t *)malloc((width - 1) * height * sizeof(uint8_t));  // Allocate new gray scale after remove seam
        int *newPriority = (int *)malloc((width - 1) * height * sizeof(int));               // Allocate new priority after remove seam

        // Find min index of last row
        int minCol = 0;         // index for remove seam
        for (int c = 1; c < width; c++) 
            if (score[(height - 1) * width + c] < score[(height - 1) * width + minCol])
                minCol = c;
        
        int minCol1 = minCol;   // index for recalculate priority

        // Trace and remove seam from last to first row
        for(int r = height - 1; r >= 0; r--) 
        {  
            // Remove seam pixel on row r by copy first and second parts (devited by seam) to new row
            // Remove from img
            memcpy(newOutPixels + r * (width - 1), tmpOutPixels + r * width, minCol * sizeof(uchar3));                                             // Copy first part
            memcpy(newOutPixels + r * (width - 1) + minCol, tmpOutPixels + r * width + minCol + 1, (width - minCol - 1) * sizeof(uchar3));         // Copy second part
            
            // Remove from gray scale
            memcpy(newGrayPixels + r * (width - 1), grayPixels + r * width, minCol * sizeof(uint8_t));                                          // Copy first part
            memcpy(newGrayPixels + r * (width - 1) + minCol, grayPixels + r * width + minCol + 1, (width - minCol - 1) * sizeof(uint8_t));      // Copy second part
            // Remove from priority, more complicated because seam's neighbor (around 3 index) have been affected
            if(minCol - 3 >= 0)                                                                                                                 // Copy first part
                memcpy(newPriority + r * (width - 1), priority + r * width, (minCol - 2) * sizeof(int));    
            if(minCol + 3 < width)                                                                                                              // Copy second part
                memcpy(newPriority + r * (width - 1) + minCol + 2, priority + r * width + minCol + 3, (width - minCol - 3) * sizeof(int));                                      
            
            // Trace up
            minCol += path[r * width + minCol];
        }
        
        width--;                                                                            // Assign 3 things to have new set with new width = (width - 1):
        uchar3 * dummyOut = tmpOutPixels; tmpOutPixels = newOutPixels; free(dummyOut);      //  + New img
        uint8_t * dummyGray = grayPixels; grayPixels = newGrayPixels; free(dummyGray);      //  + New gray scale
        int * dummyPriority = priority; priority = newPriority; free(dummyPriority);        //  + New priority
        for(int r = height - 1; r >= 0; r--)                                                //      recalculate priority at seam's neighors
        {                                          
            for(int i = -2; i < 2; i++)
                if(minCol1 + i > -1 && minCol1 + i < width)
                    priority[r * width + minCol1 + i] = computePixelPriority(grayPixels, r, minCol1 + i, width, height);
            minCol1 += path[r * (width + 1) + minCol1];
        }

        free(score);
        free(path);
    }
    
    memcpy(outPixels, tmpOutPixels, targetWidth * height * sizeof(uchar3));
    free(tmpOutPixels);
    free(grayPixels);
    free(priority);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
}

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels) 
{
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = r * width + c;
    if (r < height && c < width) 
        outPixels[idx] = 0.299f * inPixels[idx].x + 0.587f * inPixels[idx].y + 0.114f * inPixels[idx].z;
}

__constant__ int d_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int d_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void computePriorityKernel(uint8_t * inPixels, int width, int height, int * priority) 
{
    extern __shared__ uint8_t s_inPixels[];
    
    // Each block loads data from GMEM to SMEM
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

	int smemHeight = (blockDim.y + 2);
	int smemWidth = (blockDim.x + 2);
	float numBatchs = float((smemHeight * smemWidth)) / (blockDim.x * blockDim.y);

	for(int i = 0; i < numBatchs; i++)
	{
		int dest = threadIdx.x + (threadIdx.y * blockDim.x) + blockDim.x * blockDim.y * i;
		int destY = dest / smemWidth;
		int destX = dest % smemWidth;

		int srcY = destY + (blockIdx.y * blockDim.y) - 1;
		if(srcY < 0) srcY = 0;
		else if(srcY >= height) srcY = height - 1;

		int srcX = destX + (blockIdx.x * blockDim.x) - 1;
		if(srcX < 0) srcX = 0;
		else if(srcX >= width) srcX = width - 1;

		int src = srcX + (srcY * width);

		if(destY < smemHeight)
			s_inPixels[destY * smemWidth + destX] = inPixels[src];
	}

    __syncthreads();
    // ---------------------------------------

    // Each valid thread compute priority on SMEM andwrites result from SMEM to GMEM
    if (c < width && r < height) 
    {
        int x = 0, y = 0;
        for (int filterR = 0; filterR < 3; ++filterR)
            for (int filterC = 0; filterC < 3; ++filterC) 
            {
                uint8_t closest = s_inPixels[(threadIdx.y + filterR) * smemWidth + threadIdx.x + filterC];
                size_t filterIdx = filterR * 3 + filterC;
                x += closest * d_xSobel[filterIdx];
                y += closest * d_ySobel[filterIdx];
            }
    
        size_t idx = r * width + c;
        priority[idx] = abs(x) + abs(y);
    }
}

__global__ void computeSeamScoreTableKernel(int *priority, int *score, int * path, int width, int r) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < width)
    {
        int idx = r * width + c;
        int aboveIdx = (r - 1) * width + c;

        int min = score[aboveIdx];
        if (c > 0 && score[aboveIdx - 1] < min) 
        {
            min = score[aboveIdx - 1];
            path[idx] = -1;
        }
        
        if (c < width - 1 && score[aboveIdx + 1] < min) 
        {
            min = score[aboveIdx + 1];
            path[idx] = 1;
        }
        
        score[idx] = min + priority[idx];
    }
}

__global__ void removeSeamKernel(int * route, uchar3 * inPixels, uchar3 * newInpixels, uint8_t * grayPixels, uint8_t * newGrayPixels, int * priority, int * newPriority, int width, int height)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < height)
    {
        memcpy(newInpixels + r * (width - 1), inPixels + r * width, route[r] * sizeof(uchar3));
        memcpy(newInpixels + r * (width - 1) + route[r], inPixels + r * width + route[r] + 1, (width - route[r] - 1) * sizeof(uchar3));

        memcpy(newGrayPixels + r * (width - 1), grayPixels + r * width, route[r] * sizeof(uint8_t));
        memcpy(newGrayPixels + r * (width - 1) + route[r], grayPixels + r * width + route[r] + 1, (width - route[r] - 1) * sizeof(uint8_t));

        if(route[r] - 3 >= 0)
            memcpy(newPriority + r * (width - 1), priority + r * width, (route[r] - 2) * sizeof(int));
        if(route[r] + 3 < width)
            memcpy(newPriority + r * (width - 1) + route[r] + 2, priority + r * width + route[r] + 3, (width - route[r] - 3) * sizeof(int));
    }
}

__global__ void recalculatePriority(int * priority, uint8_t * gray, int * route, int width, int height)
{
    extern __shared__ uint8_t s_gray[];

    // Each block loads data from GMEM to SMEM
    int smemHeight = (blockDim.y + 2);
    int smemWidth = (blockDim.x + 2);
    float numBatchs = float((smemHeight * smemWidth)) / (blockDim.x * blockDim.y);

    for(int i = 0; i < numBatchs; i++)
    {
        int dest = threadIdx.x + (threadIdx.y * blockDim.x) + blockDim.x * blockDim.y * i;
        int destY = dest / smemWidth;
        int destX = dest % smemWidth;

        int srcY = destY + (blockIdx.y * blockDim.y) - 1;
        if(srcY < 0) srcY = 0;
        else if(srcY >= height) srcY = height - 1;

        int srcX = destX + route[srcY] - 2;
        if(srcX < 0) srcX = 0;
        else if(srcX >= width) srcX = width - 1;

        int src = srcX + (srcY * width);

        if(destY < smemHeight)
            s_gray[destY * smemWidth + destX] = gray[src];
    }

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if(r < height)
    {
        int c = threadIdx.x + route[r] - 1;
        if(c >=0 && c < width)
        {
            int x = 0, y = 0;
            for (int filterR = 0; filterR < 3; ++filterR)
                for (int filterC = 0; filterC < 3; ++filterC) 
                {
                    uint8_t closest = s_gray[(threadIdx.y + filterR) * smemWidth + threadIdx.x + filterC];
                    size_t filterIdx = filterR * 3 + filterC;
                    x += closest * d_xSobel[filterIdx];
                    y += closest * d_ySobel[filterIdx];
                }
        
            size_t idx = r * width + c;
            priority[idx] = abs(x) + abs(y);
        }
    }
}

void seamCarvingByDevice(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels, dim3 blockSize) 
{
    GpuTimer timer;
    timer.Start();

    // Allocate memory
    uchar3 *d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_priority;
    CHECK(cudaMalloc(&d_priority, width * height * sizeof(int)));
    
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    // Copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Turn input image to grayscale
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    size_t smemSize = ((blockSize.x + 2) * (blockSize.y + 2)) * sizeof(uint8_t);

    // Compute pixel priority
    computePriorityKernel<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_priority);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    dim3 edgeY(blockSize.y);
    dim3 gridSizeY((height - 1) / edgeY.x + 1);

    dim3 neighbour(4, blockSize.y);
    dim3 gridSizeNeighbour(1, (height - 1) / neighbour.y + 1);
    size_t smemSizeNeighbour = ((4 + 2) * (blockSize.y + 2)) * sizeof(uint8_t);

    while (width > targetWidth)
    {
        int * d_score;
        CHECK(cudaMalloc(&d_score, width * height * sizeof(int)));
        int * d_path;
        CHECK(cudaMalloc(&d_path, width * height * sizeof(int)));
        CHECK(cudaMemset(d_path, 0, width * height * sizeof(int)));
        CHECK(cudaMemcpy(d_score, d_priority, width * sizeof(int), cudaMemcpyDeviceToDevice));

        dim3 edgeX(blockSize.x);
        dim3 gridSizeX((width - 1) / edgeX.x + 1);

        for(int i = 1; i < height; i++)
        {
            computeSeamScoreTableKernel<<<gridSizeX, edgeX>>> (d_priority, d_score, d_path, width, i);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
        }

        // Compute min seam table
        int * score = (int *)malloc(width * height * sizeof(int));       // Dynamic score table
        int * path = (int *)malloc(width * height * sizeof(int));        // Dynamic path table

        CHECK(cudaMemcpy(score, d_score, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(path, d_path, width * height * sizeof(int), cudaMemcpyDeviceToHost));

        int * route = (int *)malloc(height * sizeof(int));
        route[height - 1] = 0;
        // Find min index of last row      
        for (int c = 1; c < width; c++) 
            if (score[(height - 1) * width + c] < score[(height - 1) * width + route[height - 1]])
                route[height - 1] = c;

        for(int i = height - 2; i >= 0; i--) 
            route[i] = route[i + 1] + path[(i + 1) * width + route[i + 1]];

        int * d_route;
        CHECK(cudaMalloc(&d_route, height * sizeof(int)));
        CHECK(cudaMemcpy(d_route, route, height * sizeof(int), cudaMemcpyHostToDevice));

        uchar3 * d_newInPixels;                                                             // Allocate new picture after remove seam
        CHECK(cudaMalloc(&d_newInPixels, (width - 1) * height * sizeof(uchar3)));
        uint8_t * d_newGrayPixels;                                                          // Allocate new gray scale after remove seam
        CHECK(cudaMalloc(&d_newGrayPixels, (width - 1) * height * sizeof(uint8_t)));
        int * d_newPriority;                                                                // Allocate new priority after remove seam
        CHECK(cudaMalloc(&d_newPriority, (width - 1) * height * sizeof(int)));

        removeSeamKernel<<<gridSizeY, edgeY>>> (d_route, d_inPixels, d_newInPixels, d_grayPixels, d_newGrayPixels, d_priority, d_newPriority, width, height);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        
        width--;                                                                                            // Assign 3 things to have new set with new width = (width - 1):
        uchar3 * dummyIn = d_inPixels; d_inPixels = d_newInPixels; CHECK(cudaFree(dummyIn));                //  + New img
        uint8_t * dummyGray = d_grayPixels; d_grayPixels = d_newGrayPixels; CHECK(cudaFree(dummyGray));     //  + New gray scale
        int * dummyPriority = d_priority; d_priority = d_newPriority; CHECK(cudaFree(dummyPriority));       //  + New priority
        
        recalculatePriority<<<gridSizeNeighbour, neighbour, smemSizeNeighbour>>> (d_priority, d_grayPixels, d_route, width, height);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        
        free(score);
        free(path);
        free(route);
        CHECK(cudaFree(d_score));
        CHECK(cudaFree(d_path));
        CHECK(cudaFree(d_route));
    }
    
    CHECK(cudaMemcpy(outPixels, d_inPixels, targetWidth * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_priority));

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use device): %f ms\n\n", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %zu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %zu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{   
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

	printDeviceInfo();

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Get the number of seam we need to remove
	int numSeamRemoved = atoi(argv[3]);
    if (checkParameter(width, height, numSeamRemoved))
        return EXIT_FAILURE; // invalid ratio
    printf("Number of seam removed: %d\n\n", numSeamRemoved);

	// Caculate the width of the result image
	int targetWidth = width - numSeamRemoved;

	// Seam carving using host
    uchar3 * correctOutPixels = (uchar3 *)malloc(targetWidth * height * sizeof(uchar3));
    seamCarvingByHost(inPixels, width, height, targetWidth, correctOutPixels);

    // Seam carving using device
    uchar3 * outPixels= (uchar3 *)malloc(targetWidth * height * sizeof(uchar3));
    dim3 blockSize(32, 32); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    } 
    seamCarvingByDevice(inPixels, width, height, targetWidth, outPixels, blockSize);
    
    // Compute mean absolute error between host result and device result
    float err = computeError(outPixels, correctOutPixels, targetWidth * height);
    printf("Error between device result and host result: %f\n", err);

    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(outPixels, targetWidth, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
	free(inPixels);
	free(correctOutPixels);
    free(outPixels);

    return 0;
}