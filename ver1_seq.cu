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

void seamCarving(uchar3 *inPixels, int width, int height, int targetWidth, uchar3* outPixels) 
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
            for(int i = 0; i < minCol; i++)
            {
                newOutPixels[i + r * (width - 1)] = tmpOutPixels[i + r * width];    // Copy first part
                newGrayPixels[i + r * (width - 1)] = grayPixels[i + r * width];     // Copy first part
            }

            for(int i = 0; i < width - minCol - 1; i++)
            {
                newOutPixels[i + r * (width - 1) + minCol] = tmpOutPixels[i + r * width + minCol + 1];    // Copy second part
                newGrayPixels[i + r * (width - 1) + minCol] = grayPixels[i + r * width + minCol + 1];     // Copy second part
            }
            // Remove from priority, more complicated because seam's neighbor (around 3 index) have been affected
            if(minCol - 3 >= 0)     
                for(int i = 0; i < minCol - 2; i++)  
                    newPriority[i + r * (width - 1)] = priority[i + r * width];                                                              // Copy first part   
            if(minCol + 3 < width)  
                for(int i = 0; i < width - minCol - 3; i++)     
                    newPriority[i + r * (width - 1) + minCol + 2] = priority[i + r * width + minCol + 3];// Copy second part                                  
            
            // Trace up
            minCol += path[r * width + minCol];
        }
        
        width--;                                                                            // Assign 3 things to have new set with new width = (width - 1):
        uchar3 * dummyOut = tmpOutPixels; tmpOutPixels = newOutPixels; free(dummyOut);            //  + New img
        uint8_t * dummyGray = grayPixels; grayPixels = newGrayPixels; free(dummyGray);      //  + New gray scale
        int * dummyPriority = priority; priority = newPriority; free(dummyPriority);        //  + New priority
        for(int r = height - 1; r >= 0; r--)                                                //      recalculate priority at seam's neighors
            for(int i = -2; i < 2; i++)
            {
                if(minCol1 + i > -1 && minCol1 < width)
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
    if (argc != 4)
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

	// Seam carving
    uchar3 * correctOutPixels = (uchar3 *)malloc(targetWidth * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, targetWidth, correctOutPixels);
    
    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(correctOutPixels, targetWidth, height, concatStr(outFileNameBase, "_host.pnm"));

    // Free memories
	free(inPixels);
	free(correctOutPixels);

    return 0;
}