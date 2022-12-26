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

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
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

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint8_t * outPixels) 
{
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
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

int computePixelPriority(uint8_t * grayPixels, int row, int col, int width, int height) 
{
    int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
    int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

    int x = 0, y = 0;
    for (int i = 0; i < 3; ++i) 
        for (int j = 0; j < 3; ++j) 
        {
            uint8_t closest = getClosest(grayPixels, row - 1 + i, col - 1 + j, width, height);
            x += closest * xSobel[i][j];
            y += closest * ySobel[i][j];
        }

    return abs(x) + abs(y);
}

void computeSeamScoreTable(int *priority, int *score, int *path, int width, int height) 
{
    for (int c = 0; c < width; ++c)
        score[c] = priority[c];
    
    for (int r = 1; r < height; ++r) 
        for (int c = 0; c < width; ++c) 
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

void seamCarving(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) 
{
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    // const int originalWidth = width;

    // Allocate memory
    int *priority = (int *)malloc(width * height * sizeof(int));
    
    uint8_t *grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
    
    // Turn input image to grayscale
    convertRgb2Gray(inPixels, width, height, grayPixels);

    // Compute pixel priority
    for (int r = 0; r < height; ++r) 
        for (int c = 0; c < width; ++c) 
            priority[r * width + c] = computePixelPriority(grayPixels, r, c, width, height);

    while (width > targetWidth) 
    {
        // Compute min seam table
        int *score = (int *)malloc(width * height * sizeof(int));       // Dynamic score table
        int *path = (int *)malloc(width * height * sizeof(int));        // Dynamic path table
        memset(path, 0, width * height);                                // Set all path to 0

        computeSeamScoreTable(priority, score, path, width, height);    // Compute seams

        // Find min index of last row
        int minCol = 0, r = height - 1;
        for (int c = 1; c < width; ++c) 
            if (score[r * width + c] < score[r * width + minCol])
                minCol = c;

    //     // Trace and remove seam from last to first row
    //     for (; r >= 0; --r) 
    //     {
    //         // Remove seam pixel on row r
    //         for (int i = minCol; i < width - 1; ++i) {
    //             outPixels[r * originalWidth + i] = outPixels[r * originalWidth + i + 1];
    //             grayPixels[r * originalWidth + i] = grayPixels[r * originalWidth + i + 1];
    //             priority[r * originalWidth + i] = priority[r * originalWidth + i + 1];
    //         }

    //         // Update priority
    //         if (r < height - 1)
    //             for (int affectedCol = 0; affectedCol < width - 1; ++affectedCol)
    //                 priority[(r + 1) * originalWidth + affectedCol] = computePixelPriority(grayPixels, r + 1, affectedCol, width - 1, height, originalWidth);

    //         // Trace up
    //         if (r > 0) 
    //         {
    //             int aboveIdx = (r - 1) * originalWidth + minCol;
    //             int min = score[aboveIdx], minColCpy = minCol;
    //             if (minColCpy > 0 && score[aboveIdx - 1] < min) {
    //                 min = score[aboveIdx - 1];
    //                 minCol = minColCpy - 1;
    //             }
    //             if (minColCpy < width - 1 && score[aboveIdx + 1] < min) {
    //                 minCol = minColCpy + 1;
    //             }
    //         }
    //     }

    //     for (int affectedCol = 0; affectedCol < width - 1; ++affectedCol) {
    //         priority[affectedCol] = computePixelPriority(grayPixels, 0, affectedCol, width - 1, height, originalWidth);
    //     }

        --width;
    }
    
    free(grayPixels);
    free(priority);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (use host): %f ms\n\n", time);
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

int main(int argc, char ** argv)
{   
    // if (argc != 4)
    // {
    //     printf("The number of arguments is invalid\n");
    //     return EXIT_FAILURE;
    // }

	// printDeviceInfo();

	// // Read input image file
	// int numChannels, width, height;
	// uchar3 * inPixels;
	// readPnm(argv[1], numChannels, width, height, inPixels);
	// printf("\nImage size (width x height): %i x %i\n", width, height);

	// // Get the number of seam we need to remove
	// int numSeamRemoved = atoi(argv[3]);
    // if (numSeamRemoved <= 0 || numSeamRemoved >= width)
    //     return EXIT_FAILURE; // invalid ratio
    // printf("Number of seam removed: %d\n\n", numSeamRemoved);

	// // Caculate the width of the result image
	// int targetWidth = width - numSeamRemoved;

	// // Seam carving
    // uchar3 * correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    // seamCarving(inPixels, width, height, targetWidth, correctOutPixels);

    return 0;
}