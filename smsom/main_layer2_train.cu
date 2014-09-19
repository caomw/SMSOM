
//cuda5.0
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
//OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//std
#include <iostream>
#include <vector>
#include <cstdlib>
//
using namespace cv;
using namespace std;

__device__ const float PI = 3.1415926;
__device__ float gaussKernel[3][3] = {1/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0, 2/16.0, 1/16.0, 2/16.0, 1/16.0};

__device__ int mi[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
__device__ int mj[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

__device__ int xlu[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};
__device__ int xu[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int xru[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xr[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xrd[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xd[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int xld[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};
__device__ int xl[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};

__device__ int ylu[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yu[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yru[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yr[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int yrd[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yd[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yld[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yl[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ float distance(float h1, float s1, float v1, 
	float h2, float s2, float v2){
		return sqrtf(pow(s1*v1*cos(h1*PI/180) - s2*v2*cos(h2*PI/180), 2) +
			pow(s1*v1*sin(h1*PI/180) - s2*v2*sin(h2*PI/180), 2) +
			pow(v1 - v2, 2));
}

__device__ bool shadowRemove(float hi, float si, float vi,
	float hm, float sm, float vm){
		return ( (vi/vm<1) && (vi/vm>0.7) && (si-sm<0.1) && (fabs(hi-hm)<10) );
}

//
__global__ void initLayer(float* input, float* output, int width){
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;//thread index

	for (int j=0; j<3; ++j){
		for (int i=0; i<3; ++i){
			output[(y*3+j)*width*3+(x*3+i)] = input[y*width+x];
		}
	}
}

//foreground detection
__global__ void compete(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV, 
	bool* match, int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		//used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = modelH[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = modelS[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = modelV[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int index = 0;
		int i2 = 0;
		float min = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist <= min){
				min = dist; 
				index = i2;
			}
		}

		for (int j3 = 0; j3 < 3; ++j3){
			for (int i3 = 0; i3 < 3; ++i3){
				match[(y*3+j3)*width*3+(x*3+i3)] = false;
			}
		}
		match[(y*3+mj[index])*width*3+(x*3+mi[index])] = true;
}

__global__ void competeWithFilter(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue,
	bool* match, int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		//used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		for (int j3 = 0; j3 < 3; ++j3){
			for (int i3 = 0; i3 < 3; ++i3){
				match[(y*3+j3)*width*3+(x*3+i3)] = false;
			}
		}

		if( max >= maxValue[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					pointModel[j*3+i][0] = model2H[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][1] = model2S[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][2] = model2V[(y*3+j)*width*3+(x*3+i)];
				}
			}

			int index = 0;
			int i2 = 0;
			float min = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

			for (int i2 = 1; i2 < 3*3; ++i2){ 
				float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
				if (dist <= min){
					min = dist; 
					index = i2;
				}
			}
			match[(y*3+mj[index])*width*3+(x*3+mi[index])] = true;
		}
}

//update the background model
__global__ void cooperate(float* modelH, float* modelS, float* modelV, 
	float* backupH, float* backupS, float* backupV,
	float* frameH, float* frameS, float* frameV,
	bool* match, 
	int width, int height, float alpha){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	int m = 0;
	for(int j = 0; j < 3; ++j){
		for(int i = 0; i < 3; ++i){
			m = j*3+i;
			//center
			if(match[(y*3+j)*width*3+(x*3+i)] == true){
				modelH[(y*3+j)*width*3+(x*3+i)] = 
					(1-alpha*gaussKernel[1][1])*backupH[(y*3+j)*width*3+(x*3+i)]
				+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
				modelS[(y*3+j)*width*3+(x*3+i)] =
					(1-alpha*gaussKernel[1][1])*backupS[(y*3+j)*width*3+(x*3+i)] 
				+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
				modelV[(y*3+j)*width*3+(x*3+i)] =
					(1-alpha*gaussKernel[1][1])*backupV[(y*3+j)*width*3+(x*3+i)] + 
					alpha*gaussKernel[1][1]*(frameV[y*width+x]);
			}
			//left up
			if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
				match[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupH[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
					+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupS[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
					+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupV[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
					+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
			}
			//up
			if (  (y+yu[m])>=0 && 
				match[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupH[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupS[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupV[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
			}
			//right up
			if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
				match[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][0])*backupH[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
					+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] =
						(1-alpha*gaussKernel[2][0])*backupS[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
					+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][0])*backupV[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
			}
			//right
			if (  (x+xr[m])<=width && 
				match[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupH[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupS[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupV[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
			}
			//right down
			if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
				match[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupH[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupS[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupV[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
			}
			//down
			if (  (y+yd[m])>=height && 
				match[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupH[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupS[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupV[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
			}
			//left down7
			if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
				match[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] =
						(1-alpha*gaussKernel[0][2])*backupH[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][2])*backupS[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][2])*backupV[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
			}
			//left
			if (  (x+xl[m])>=0 && 
				match[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupH[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupS[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupV[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
			}
		}
	}
}

__global__ void cooperateWithFilter(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V, 
	float* backup2H, float* backup2S, float* backup2V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue,
	bool* match, 
	int width, int height, float alpha){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		//used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		if( max >= maxValue[y*width+x] ){
			int m = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					m = j*3+i;
					//center
					if(match[(y*3+j)*width*3+(x*3+i)] == true){
						model2H[(y*3+j)*width*3+(x*3+i)] = 
							(1-alpha*gaussKernel[1][1])*backup2H[(y*3+j)*width*3+(x*3+i)]
						+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
						model2S[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2S[(y*3+j)*width*3+(x*3+i)] 
						+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
						model2V[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2V[(y*3+j)*width*3+(x*3+i)] + 
							alpha*gaussKernel[1][1]*(frameV[y*width+x]);
					}
					//left up
					if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
						match[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
							+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
					}
					//up
					if (  (y+yu[m])>=0 && 
						match[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
					}
					//right up
					if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
						match[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[2][0])*backup2S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
							+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
					}
					//right
					if (  (x+xr[m])<=width && 
						match[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
					}
					//right down
					if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
						match[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
					}
					//down
					if (  (y+yd[m])>=height && 
						match[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
					}
					//left down7
					if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
						match[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[0][2])*backup2H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
					}
					//left
					if (  (x+xl[m])>=0 && 
						match[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
					}
				}
			}
		}
}

__global__ void initMean(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		//used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				meanDistance[(y*3+j)*width*3+(x*3+i)] = 
					distance(pointFrame[0], pointFrame[1], pointFrame[2],
					modelH[(y*3+j)*width*3+(x*3+i)], 
					modelS[(y*3+j)*width*3+(x*3+i)],
					modelV[(y*3+j)*width*3+(x*3+i)]);
			}
		}
}

__global__ void meanSum(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		//used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				meanDistance[(y*3+j)*width*3+(x*3+i)] = (
					meanDistance[(y*3+j)*width*3+(x*3+i)] +
					distance(pointFrame[0], pointFrame[1], pointFrame[2],
					modelH[(y*3+j)*width*3+(x*3+i)], 
					modelS[(y*3+j)*width*3+(x*3+i)],
					modelV[(y*3+j)*width*3+(x*3+i)])
					)/2;
			}
		}
}

__global__ void calculateThreshold(float* meanValue, float* maxValue, float* thresholdValue,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		float tempMax = meanValue[(y*3)*width*3+(x*3)];
		float tempThreshold = 0;
		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				if( meanValue[(y*3+j)*width*3+(x*3+i)]>=tempMax )
					tempMax = meanValue[(y*3+j)*width*3+(x*3+i)];
				tempThreshold += meanValue[(y*3+j)*width*3+(x*3+i)];
			}
		}

		maxValue[y*width+x] = tempMax;
		thresholdValue[y*width+x] = tempThreshold/9;
}

int thresholdK = 200;
float yipuxilu1 = 0.1;
float yipuxilu2 = 0.03;
float c1 = 1;
float c2 = 0.05;
float alphaLearning = c1*4; // c1/max weight of the Gaussian kernel
float alpha2 = c2*4; // c2/max weight of the Gaussian kernel
int startFrame = 2, endFrame = 799;

int main(){
	char path[200] = "E:\\数据备份, 以前的研究等\\data收集\\CDnet\\CDnet\\dataset\\dynamicBackground\\overpass\\input\\in%06d.jpg";
	char fileName[200];

	//
	Mat frame;
	sprintf(fileName, path, 1);
	frame = imread(fileName, CV_LOAD_IMAGE_COLOR);
	int width = frame.cols;
	int height = frame.rows;

	Mat frameFloat;
	Mat frameFloat2;
	frameFloat.create(height, width, CV_32FC3);
	frameFloat2.create(height, width, CV_32FC3);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);

	vector<Mat> input(3);
	input[0].create(height, width, CV_32FC1);
	input[1].create(height, width, CV_32FC1);
	input[2].create(height, width, CV_32FC1);
	split(frameFloat2, input);

	vector<float*> gpuInput(3);
	vector<float*> gpuLayer1(3);
	vector<float*> gpuLayer1Backup(3);
	bool* gpuMatch1;
	float* gpuOutput;
	float* gpuOutputBackup;

	Mat output;
	output.create(height, width, CV_32FC1);

	for(int i = 0; i < 3; ++i){
		cudaMalloc((void**)&gpuInput[i], width*height*sizeof(float));
		cudaMalloc((void**)&gpuLayer1[i], width*height*3*3*sizeof(float));
		cudaMalloc((void**)&gpuLayer1Backup[i], width*height*3*3*sizeof(float));
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaMalloc((void**)&gpuMatch1, width*height*3*3*sizeof(bool));
	cudaMalloc((void**)&gpuOutput, width*height*sizeof(float));
	cudaMalloc((void**)&gpuOutputBackup, width*height*sizeof(float));

	dim3 grid(width/16, height/16, 1);
	dim3 block(16, 16, 1);

	//Stacked Multi-layer Self Organizing Map (in this code, 2 layers)

	//initialize layer 1
	for(int i = 0; i < 3; ++i){
		initLayer<<<grid, block>>>(gpuInput[i], gpuLayer1[i], width);
	}

	//train layer 1
	cout<<"start training layer 1 ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer1Backup[j], gpuLayer1[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		compete<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMatch1, width);
		cooperate<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer1Backup[0], gpuLayer1Backup[1], gpuLayer1Backup[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMatch1,
			width, height, alphaLearning);
	}

	//initialize layer 2
	float* gpuMeanDistance1;
	float* gpuMaxDistance1;
	float* gpuThreshold1;
	cudaMalloc((void**)&gpuMeanDistance1, width*height*3*3*sizeof(float));
	cudaMalloc((void**)&gpuMaxDistance1, width*height*sizeof(float));
	cudaMalloc((void**)&gpuThreshold1, width*height*sizeof(float));
	
	//first frame
	sprintf(fileName, path, 1);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	initMean<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
		gpuInput[0], gpuInput[1], gpuInput[2],
		gpuMeanDistance1, width);

	cout<<"calculate the thresholds for detection and layer 2 input ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
		}

		meanSum<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMeanDistance1, width);
	}

	//
	calculateThreshold<<<grid, block>>>(gpuMeanDistance1, 
		gpuMaxDistance1, gpuThreshold1, width);

	//train layer 2
	vector<float*> gpuLayer2(3);
	vector<float*> gpuLayer2Backup(3);
	bool* gpuMatch2;
	for (int i = 0; i < 3; ++i){
		cudaMalloc((void**)&gpuLayer2[i], width*height*3*3*sizeof(float));
		cudaMalloc((void**)&gpuLayer2Backup[i], width*height*3*3*sizeof(float));
	}
	cudaMalloc((void**)&gpuMatch2, width*height*3*3*sizeof(bool));

	//first frame
	sprintf(fileName, path, 1);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 3; ++i){
		initLayer<<<grid, block>>>(gpuInput[i], gpuLayer2[i], width);
	}

	cout<<"start training layer 2 ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer2Backup[j], gpuLayer2[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		competeWithFilter<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMatch2, width);
		cooperateWithFilter<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer2Backup[0], gpuLayer2Backup[1], gpuLayer2Backup[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMatch2,
			width, height, alphaLearning);
	}

	for(int i = 0; i < 3; ++i){
		Mat outputTemp;
		outputTemp.create(height*3, width*3, CV_32FC1);
		cudaMemcpy(outputTemp.data, gpuLayer2[i], width*height*3*3*sizeof(float), cudaMemcpyDeviceToHost);
		namedWindow("layer2", 1);
		imshow("layer2", outputTemp/360);
		waitKey(0);
	}


	//DEBUG
	//cudaMemcpy(output.data, gpuMaxDistance1, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//namedWindow("max", 1);
	//imshow("max", output);
	//waitKey(0);

	//cudaMemcpy(output.data, gpuThreshold1, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//namedWindow("threshold", 1);
	//imshow("threshold", output);
	//waitKey(0);

	return 0;
}