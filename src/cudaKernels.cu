#define CONV_SOBEL_SIZE 3
#define CONV_GAUSSIAN_SIZE 5

__constant__ char SOBELX[CONV_SOBEL_SIZE*CONV_SOBEL_SIZE] = {-1,0,1,-2,0,2,-1,0,1};
__constant__ char SOBELY[CONV_SOBEL_SIZE*CONV_SOBEL_SIZE] = {1,2,1,0,0,0,-1,-2,-1};

__constant__ char GAUSSIAN[CONV_GAUSSIAN_SIZE*CONV_GAUSSIAN_SIZE] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};

__global__ void rgb_to_grayscale( unsigned char* imageInput, 
    unsigned char* imageOutput, 
    int width,
    int height,
    int yIndexorWidthStep,
    int grayWidthStep)
{
    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    //Only valid threads
    if((xIndex<width) && (yIndex<height))
    {
        const int yIndexor_tid = yIndex * yIndexorWidthStep + (3 * xIndex);

        //Location of gray pixel in imageOutput
        const int gray_tid  = yIndex * grayWidthStep + xIndex;

        const unsigned char blue	= imageInput[yIndexor_tid];
        const unsigned char green	= imageInput[yIndexor_tid + 1];
        const unsigned char red		= imageInput[yIndexor_tid + 2];

        //same weights as cv::COLOR_RGB2GRAY
        const float gray = red * 0.299f + green * 0.587f + blue * 0.114f;

        imageOutput[gray_tid] = static_cast<unsigned char>(gray);
    }
}

__device__ unsigned char normalize(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

//funcion global de filtro de sobel
__global__ void sobel(unsigned char *imageInput, 
    unsigned char *imageOutput,
    int width, 
    int height,
    unsigned int maskWidth){

    //2D Index of current thread
    unsigned int xIndex = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int yIndex = blockIdx.x*blockDim.x+threadIdx.x;

    int temp_sobel_x = 0;
    int temp_sobel_y = 0;
    
    int start_xIndex = xIndex - (maskWidth/2);
    int start_yIndex = yIndex - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((start_yIndex + j >=0 && start_yIndex + j < width) \
                    &&(start_xIndex + i >=0 && start_xIndex + i < height)){
                        temp_sobel_x += imageInput[(start_xIndex + i)*width+(start_yIndex + j)] * SOBELX[i*maskWidth+j];
                        temp_sobel_y += imageInput[(start_xIndex + i)*width+(start_yIndex + j)] * SOBELY[i*maskWidth+j];
            }
        }
    }
    temp_sobel_x = normalize(temp_sobel_x);
    temp_sobel_y = normalize(temp_sobel_y);

    imageOutput[xIndex*width+yIndex] = (int)sqrt((float)(temp_sobel_y*temp_sobel_y)+(temp_sobel_x*temp_sobel_x));
}

//funcion global de filtro de sobel
__global__ void gaussian(unsigned char *imageInput, 
    unsigned char *imageOutput,
    int width, 
    int height,
    unsigned int maskWidth){

    //2D Index of current thread
    unsigned int xIndex = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int yIndex = blockIdx.x*blockDim.x+threadIdx.x;

    float temp_gaussian = 0;

    int start_xIndex = xIndex - (maskWidth/2);
    int start_yIndex = yIndex - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((start_yIndex + j >=0 && start_yIndex + j < width) \
                    &&(start_xIndex + i >=0 && start_xIndex + i < height)){
                        temp_gaussian += (float)imageInput[(start_xIndex + i)*width+(start_yIndex + j)] * ((float)GAUSSIAN[i*maskWidth+j]/(float)255);
            }
        }
    }

    imageOutput[xIndex*width+yIndex] = normalize(temp_gaussian);;
}


extern "C" void cuda_grayscale(unsigned char* imageInput, 
    unsigned char* imageOutput, 
    int width,
    int height,
    int yIndexorWidthStep,
    int grayWidthStep, 
    dim3 grid, 
    dim3 block_size)
{
	rgb_to_grayscale <<< grid, block_size >>> ((unsigned char*)imageInput,(unsigned char*)imageOutput, width, height,yIndexorWidthStep,grayWidthStep);
}


extern "C" void cuda_Sobel(unsigned char *imageInput,
    unsigned char *imageOutput, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 grid, 
    dim3 block_size)
{
	sobel <<< grid, block_size >>> ((unsigned char*)imageInput,(unsigned char*)imageOutput, width, height,maskWidth);
}


extern "C" void cuda_Gaussian(unsigned char *imageInput,
    unsigned char *imageOutput, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 grid, 
    dim3 block_size)
{
	gaussian <<< grid, block_size >>> ((unsigned char*)imageInput,(unsigned char*)imageOutput, width, height,maskWidth);
}