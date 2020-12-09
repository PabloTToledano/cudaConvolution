#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3

__constant__ char M[MASK_WIDTH*MASK_WIDTH];

using namespace cv;

//funcion device
__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

//funcion global de filtro de sobel
__global__ void sobelFilter(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}

//funcion global de escala de grises
__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

//MENU
int main(int argc, char **argv){
    cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1};
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
  	
    //creamos la matriz de la imagen de entrada
    Mat image;
    //llamamos la imagen
    image = imread("./inputs/img5.jpg", 1);
  	
    //algoritmo secuencial con OpenCV
    start = clock();
    Mat gray_image_opencv, grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    end = clock();
    //fin algoritmo secuencial con OpenCV
  
    Size s = image.size();
		
    //inicializamos variables
    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;
		
    //Reserva de Memoria para d_dataRawImage
    dataRawImage = (unsigned char*)malloc(size);
    error = cudaMalloc((void**)&d_dataRawImage,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }

    //Reserva de Memoria para d_imageOutput
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }

    //Reserva de Memoria para d_sobelOutput
    error = cudaMalloc((void**)&d_sobelOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_sobelOutput\n");
        exit(-1);
    }

    dataRawImage = image.data;
		
    //Algoritmo Paralelo con CUDA
    startGPU = clock();

    error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }

    error = cudaMemcpyToSymbol(M,h_M,sizeof(char)*MASK_WIDTH*MASK_WIDTH);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_M a d_M \n");
        exit(-1);
    }

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    //llamamos la funcion de escala de grises
    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    //Sincronizamos
    cudaDeviceSynchronize();
    //llamamos la funcion de filtro de sobel
    sobelFilter<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_sobelOutput);
    cudaMemcpy(h_imageOutput,d_sobelOutput,sizeGray,cudaMemcpyDeviceToHost);

    endGPU = clock();
    //fin algoritmo Paralelo con CUDA
  	
    //creamos la matriz de la imagen en sobel
    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

  
    //imprimir las imagenes en full color, escala de grises o filtro de sobel(secuencial o paralelo)	
			
    //imwrite("./outputs/1088012385.png",image);		//full color
    //imwrite("./outputs/1088012385.png",gray_image_opencv);	//grises secuencial
    //imwrite("./outputs/1088012385.png",grad_x);		//sobel secuencial
      imwrite("./outputs/1088012385.png",gray_image);		//sobel del paralelo
		
  
    //imprimir tiempos de ejecucion
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo Paralelo CUDA: %.10f\n",gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo Secuencial OpenCV: %.10f\n",cpu_time_used);
    printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    //limpiar memoria
    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(M);
    cudaFree(d_sobelOutput);
    return 0;
}