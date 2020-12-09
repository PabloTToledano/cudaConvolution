#define CONV_KERNEL_WIDTH 3

__constant__ char SOBELX[CONV_KERNEL_WIDTH*CONV_KERNEL_WIDTH] = {-1,0,1,-2,0,2,-1,0,1};
__constant__ char SOBELY[CONV_KERNEL_WIDTH*CONV_KERNEL_WIDTH] = {1,2,1,0,0,0,-1,-2,-1};

__global__ void rgb_to_grayscale( unsigned char* input, 
    unsigned char* output, 
    int width,
    int height,
    int colorWidthStep,
    int grayWidthStep)
{
    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    //Only valid threads perform memory I/O
    if((xIndex<width) && (yIndex<height))
    {
        //Location of colored pixel in input
        const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

        //Location of gray pixel in output
        const int gray_tid  = yIndex * grayWidthStep + xIndex;

        const unsigned char blue	= input[color_tid];
        const unsigned char green	= input[color_tid + 1];
        const unsigned char red		= input[color_tid + 2];

        const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

        output[gray_tid] = static_cast<unsigned char>(gray);
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

    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int temp_sobel_x = 0;
    int temp_sobel_y = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                        temp_sobel_x += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * SOBELX[i*maskWidth+j];
                        temp_sobel_y += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * SOBELY[i*maskWidth+j];
            }
        }
    }
    temp_sobel_x = normalize(temp_sobel_x);
    temp_sobel_y = normalize(temp_sobel_y);

    imageOutput[row*width+col] = (int)sqrt((float)(temp_sobel_y*temp_sobel_y)+(temp_sobel_x*temp_sobel_x));
}


extern "C" void cuda_grayscale(unsigned char* input, 
    unsigned char* output, 
    int width,
    int height,
    int colorWidthStep,
    int grayWidthStep, 
    dim3 blocks, 
    dim3 block_size)
{
	rgb_to_grayscale <<< blocks, block_size >>> ((unsigned char*)input,(unsigned char*)output, width, height,colorWidthStep,grayWidthStep);
}


extern "C" void cuda_Sobel(unsigned char *input,
    unsigned char *output, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 blocks, 
    dim3 block_size)
{
	sobel <<< blocks, block_size >>> ((unsigned char*)input,(unsigned char*)output, width, height,maskWidth);
}