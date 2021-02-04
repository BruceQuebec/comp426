#include <windows.h>
#include <thread>
#include <iostream>
#include <unordered_map>
#include <tbb/tbb.h>
#include <stdio.h>
#include <math.h>
#include "../CellGrowth_CUDA/Dependencies/glew/glew.h"
#include "../CellGrowth_CUDA/Dependencies/freeglut/freeglut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "angle_bracket_replace_macro.h"

using namespace std;
using namespace tbb;

//global variables
const int WIDTH = 1024, HEIGHT = 768;
bool flag = true;

/*
*	data model
*/
int cell[WIDTH][HEIGHT] = { {0} };
int medDir[WIDTH][HEIGHT] = { {0} };

int host_cell[WIDTH*HEIGHT] = { 0 };
int host_medDir[WIDTH*HEIGHT] = { 0 };

/*
*	viewer methods
*/
void init() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glLoadIdentity();
	//gluOrtho2D(-0.5f, WIDTH - 0.5f, -0.5f, HEIGHT - 0.5f);
	gluOrtho2D(-0.5f, 400 - 0.5f, -0.5f, 300 - 0.5f);
}

void draw(GLfloat red, GLfloat green, GLfloat blue, int x, int y) {
	glPointSize(3.0f);
	glColor3f(red, green, blue);
	glBegin(GL_POINTS);
	glVertex2i(x, y);
	glEnd();
}

void update(int value) {
	glutPostRedisplay();
	glutTimerFunc(10.00, update, 0);
}


/*
*	controller methods
*/
void setupCUDA(int x, int y, int m) {
	srand(time(NULL));
	rand(); rand(); rand();
	int h = (m * x) + 2;
	int w = (m * y) + 2;

	for (int i = 0; i < 50; i++) {
		int center_x;
		int center_y;
		do {
			center_x = ((h - x) + 1) + rand() / (RAND_MAX / ((h - 2) - ((h - x) + 1)));
			center_y = ((w - y) + 1) + rand() / (RAND_MAX / ((w - 2) - ((w - y) + 1)));
		} while (host_cell[center_x * HEIGHT + center_y] != 0);
		host_cell[center_x * HEIGHT + center_y] = 9;

		host_cell[center_x * HEIGHT + center_y + 1] = 3;
		host_cell[(center_x - 1) * HEIGHT + center_y + 1] = 3;
		host_cell[(center_x - 1) * HEIGHT + center_y] = 3;
		host_cell[(center_x - 1) * HEIGHT + center_y - 1] = 3;
		host_cell[center_x * HEIGHT + center_y - 1] = 3;
		host_cell[(center_x + 1) * HEIGHT + center_y - 1] = 3;
		host_cell[(center_x + 1) * HEIGHT + center_y] = 3;
		host_cell[(center_x + 1) * HEIGHT + center_y + 1] = 3;

		host_medDir[center_x * HEIGHT + center_y + 1] = 1;	//up
		host_medDir[(center_x - 1) * HEIGHT + center_y + 1] = 2; //left_up
		host_medDir[(center_x - 1) * HEIGHT + center_y] = 3;	//left;
		host_medDir[(center_x - 1) * HEIGHT + center_y - 1] = 4; //left_down;
		host_medDir[center_x * HEIGHT + center_y - 1] = 5; //down;
		host_medDir[(center_x + 1) * HEIGHT + center_y - 1] = 6;  //right_down
		host_medDir[(center_x + 1) * HEIGHT + center_y] = 7; //right
		host_medDir[(center_x + 1) * HEIGHT + center_y + 1] = 8;  //right_up;
	}
	for (int i = (h - x); i < h; i++) {
		for (int j = (w - y); j < w; j++) {
			//Initialize each pixel with an arbitry alive/dead value.
			if (host_cell[i * HEIGHT + j] != 3) {
				host_cell[i * HEIGHT + j] = ((rand() % 4) == 0) ? 1 : 2;
			}
		}
	}
}

__device__ void medCellMoveCUDA(int o_x, int o_y, int t_x, int t_y, int direction, int* dev_cell, int* dev_medDir, int num_per_row) {
	int typeTemp = typeTemp = dev_cell[t_x * num_per_row + t_y];
	dev_cell[o_x * num_per_row + o_y] = typeTemp;
	dev_cell[t_x * num_per_row + t_y] = 3;
	dev_medDir[o_x * num_per_row + o_y] = 0;
	dev_medDir[t_x * num_per_row + t_y] = direction;
}

//Check status of individual cell and apply the game rules.
__device__ void checkStatusCUDA(int* dev_cell, int cur_row, int cur_column, int num_per_row, int* state) {
	int cancerNeighbours = 0;
	int liveNeighbours = 0;
	int medNeighbours = 0;

	int upper_row = cur_row - 1 < 0 ? 0 : cur_row - 1;
	int lower_row = cur_row + 1 > 1023 ? 1023 : cur_row + 1;
	int left_column = cur_column - 1 < 0 ? 0 : cur_column - 1;
	int right_column = cur_column + 1 > 767 ? 767 : cur_column + 1;
	for (int i = upper_row; i <= lower_row; i++) {
		
		if (dev_cell[i*num_per_row + left_column] == 1 && left_column!=cur_column) {
			cancerNeighbours++;
		}
		else if (dev_cell[i*num_per_row + left_column] == 2 && left_column != cur_column) {
			liveNeighbours++;
		}
		else if (dev_cell[i*num_per_row + left_column] == 3 && left_column != cur_column) {
			medNeighbours++;
		}
		if (dev_cell[i*num_per_row + right_column] == 1 && right_column!= cur_column) {
			cancerNeighbours++;
		}
		else if (dev_cell[i*num_per_row + right_column] == 2 && right_column != cur_column) {
			liveNeighbours++;
		}
		else if (dev_cell[i*num_per_row + right_column] == 3 && right_column != cur_column) {
			medNeighbours++;
		}
	}
	if (dev_cell[cur_row*num_per_row + cur_column] != 0) {
		if (dev_cell[upper_row*num_per_row + cur_column] == 1 && upper_row!=cur_row) {
			cancerNeighbours++;
		}
		else if (dev_cell[upper_row*num_per_row + cur_column] == 2 && upper_row != cur_row) {
			liveNeighbours++;
		}
		else if (dev_cell[upper_row*num_per_row + cur_column] == 3 && upper_row != cur_row) {
			medNeighbours++;
		}

		if (dev_cell[lower_row*num_per_row + cur_column] == 1 && lower_row!=cur_row) {
			cancerNeighbours++;
		}
		else if (dev_cell[lower_row*num_per_row + cur_column] == 2 && lower_row != cur_row) {
			liveNeighbours++;
		}
		else if (dev_cell[lower_row*num_per_row + cur_column] == 3 && lower_row != cur_row) {
			medNeighbours++;
		}

		if (dev_cell[cur_row*num_per_row + cur_column] == 1 && medNeighbours >= 3) {
			*state = 2;
		}
		else if (dev_cell[cur_row*num_per_row + cur_column] == 2 && cancerNeighbours >= 5) {
			*state = 1;
		}
		else
			*state = dev_cell[cur_row*num_per_row + cur_column];
	}
	else {
		if (liveNeighbours == 3) {
			*state = 2;
		}
	}
}

void medMultiInjectionCUDA() {
	//variable indicating how many groups of medicine should be injected at a time
	int num_medicines_group = rand() % (15) + 1;
	int inject_area_width = 400;
	int inject_area_height = 300;
	int h = (1 * inject_area_width) + 2;
	int w = (1 * inject_area_height) + 2;
	bool ifInjected[WIDTH*HEIGHT] = { false };

	parallel_for(blocked_range<size_t>(0, num_medicines_group), [&](blocked_range<size_t> & a) {
		for (size_t i = a.begin(); i != a.end(); ++i) {
			//variable indicating the number of medicine cell which should be line up at the border (not including 4 corner)
			//also variable indicating how many cells should the medicine cell be placed far away from center 
			int num_medicine_factor = rand() % (15) + 1;
			//variable indicating the total number of medicine for a single group of medicine cells in one step of iteration
			int num_medicine = num_medicine_factor * 4 + 4;

			int center_x;
			int center_y;
			do {
				center_x = ((h - inject_area_width) + num_medicine_factor) + rand() / (RAND_MAX / ((h - 2) - ((h - inject_area_width) + num_medicine_factor)));
				center_y = ((w - inject_area_height) + num_medicine_factor) + rand() / (RAND_MAX / ((w - 2) - ((w - inject_area_height) + num_medicine_factor)));
			} while (ifInjected[center_x*HEIGHT + center_y] == true);
			ifInjected[center_x*HEIGHT + center_y] = true;

			parallel_for(blocked_range<size_t>(center_x - (int)num_medicine_factor / 2, center_x + (int)num_medicine_factor / 2 + 1), [&](blocked_range<size_t>& r) {
				for (int j = r.begin(); j != r.end(); ++j) {
					host_cell[j*HEIGHT + center_y + num_medicine_factor] = 3;
					host_cell[j*HEIGHT + center_y - num_medicine_factor] = 3;
					host_medDir[j*HEIGHT + center_y + num_medicine_factor] = 1;	//up
					host_medDir[j*HEIGHT + center_y - num_medicine_factor] = 5; //down;
				}
			});

			parallel_for(blocked_range<size_t>(center_y - (int)num_medicine_factor / 2, center_y + (int)num_medicine_factor / 2 + 1), [&](blocked_range<size_t>& r) {
				for (int j = r.begin(); j != r.end(); ++j) {
					host_cell[(center_x - num_medicine_factor)*HEIGHT + j] = 3;
					host_cell[(center_x + num_medicine_factor)*HEIGHT + j] = 3;
					host_medDir[(center_x - num_medicine_factor)*HEIGHT + j] = 3;	//left
					host_medDir[(center_x + num_medicine_factor)*HEIGHT + j] = 7; //right;
				}
			});

			host_cell[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor] = 3;
			host_cell[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor] = 3;
			host_cell[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor] = 3;
			host_cell[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor] = 3;
			host_medDir[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor] = 2; //left_up
			host_medDir[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor] = 4; //left_down;
			host_medDir[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor] = 6;  //right_down
			host_medDir[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor] = 8;  //right_up;
		}
	});
}

void medInjectionCUDA(GLdouble worldX, GLdouble worldY) {
	int center_x = (int)worldX;
	int center_y = (int)worldY;

	host_cell[center_x * HEIGHT + center_y + 1] = 3;
	host_cell[(center_x - 1) * HEIGHT + center_y + 1] = 3;
	host_cell[(center_x - 1) * HEIGHT + center_y] = 3;
	host_cell[(center_x - 1) * HEIGHT + center_y - 1] = 3;
	host_cell[center_x *HEIGHT + center_y - 1] = 3;
	host_cell[(center_x + 1) * HEIGHT + center_y - 1] = 3;
	host_cell[(center_x + 1) * HEIGHT + center_y] = 3;
	host_cell[(center_x + 1) * HEIGHT + center_y + 1] = 3;

	host_medDir[center_x * HEIGHT + center_y + 1] = 1;	//up
	host_medDir[(center_x - 1) *HEIGHT + center_y + 1] = 2; //left_up
	host_medDir[(center_x - 1) * HEIGHT + center_y] = 3;	//left;
	host_medDir[(center_x - 1) * HEIGHT + center_y - 1] = 4; //left_down;
	host_medDir[center_x * HEIGHT + center_y - 1] = 5; //down;
	host_medDir[(center_x + 1) * HEIGHT + center_y - 1] = 6;  //right_down
	host_medDir[(center_x + 1) * HEIGHT + center_y] = 7; //right
	host_medDir[(center_x + 1) * HEIGHT + center_y + 1] = 8;  //right_up;
}

void OnMouseClick(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		GLint viewport[4]; //var to hold the viewport info
		GLdouble modelview[16]; //var to hold the modelview info
		GLdouble projection[16]; //var to hold the projection matrix info
		GLfloat winX, winY, winZ; //variables to hold screen x,y,z coordinates
		GLdouble worldX, worldY, worldZ; //variables to hold world x,y,z coordinates
		glGetDoublev(GL_MODELVIEW_MATRIX, modelview); //get the modelview info
		glGetDoublev(GL_PROJECTION_MATRIX, projection); //get the projection matrix info
		glGetIntegerv(GL_VIEWPORT, viewport); //get the viewport info
		winX = (float)x;
		winY = (float)viewport[3] - (float)y;
		winZ = 0;
		//get the world coordinates from the screen coordinates
		gluUnProject(winX, winY, winZ, modelview, projection, viewport, &worldX, &worldY, &worldZ);
		medInjectionCUDA(worldX, worldY);
		medMultiInjectionCUDA();
		//cout << "worldX: " << worldX << "worldY: " << worldY << endl;
	}
}

//Display individual pixels.
__global__ void cellHandle(int *dev_cell, int* dev_medDir, int num_rows, int num_per_row) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int cur_row = (int) idx / num_per_row;
	int cur_column = idx % num_per_row;

	int state;
	checkStatusCUDA(dev_cell, cur_row, cur_column, num_per_row, &state);

	if (state == 2 && dev_cell[cur_row*num_per_row + cur_column] == 1) {
		if (cur_column + 1 < HEIGHT - 5 && dev_cell[cur_row*num_per_row + cur_column + 1] == 3) {
			dev_cell[cur_row*num_per_row + cur_column + 1] = 2;
			dev_medDir[cur_row*num_per_row + cur_column + 1] = 0;
		}
		if (cur_row - 1 >= 5 && cur_column + 1 < HEIGHT - 5 && dev_cell[(cur_row - 1)*num_per_row + cur_column + 1] == 3) {
			dev_cell[(cur_row - 1)*num_per_row + cur_column + 1] = 2;
			dev_medDir[(cur_row - 1)*num_per_row + cur_column + 1] = 0;
		}
		if (cur_row - 1 >= 5 && dev_cell[(cur_row - 1)*num_per_row + cur_column] == 3) {
			dev_cell[(cur_row - 1)*num_per_row + cur_column] = 2;
			dev_medDir[(cur_row - 1)*num_per_row + cur_column] = 0;
		}
		if (cur_row - 1 >= 5 && cur_column - 1 >= 5 && dev_cell[(cur_row - 1)*num_per_row + cur_column - 1] == 3) {
			dev_cell[(cur_row - 1)*num_per_row + cur_column - 1] = 2;
			dev_medDir[(cur_row - 1)*num_per_row + cur_column - 1] = 0;
		}
		if (cur_column - 1 >= 5 && dev_cell[cur_row*num_per_row + cur_column - 1] == 3) {
			dev_cell[cur_row*num_per_row + cur_column - 1] = 2;
			dev_medDir[cur_row*num_per_row + cur_column - 1] = 0;
		}
		if (cur_row + 1 < WIDTH - 5 && cur_column - 1 >= 5 && dev_cell[(cur_row + 1)*num_per_row + cur_column - 1] == 3) {
			dev_cell[(cur_row + 1)*num_per_row + cur_column - 1] = 2;
			dev_medDir[(cur_row + 1)*num_per_row + cur_column - 1] = 0;
		}
		if (cur_row + 1 < WIDTH - 5 && dev_cell[(cur_row + 1)*num_per_row + cur_column] == 3) {
			dev_cell[(cur_row + 1)*num_per_row + cur_column] = 2;
			dev_medDir[(cur_row + 1)*num_per_row + cur_column] = 0;
		}
		if (cur_row + 1 < WIDTH - 5 && cur_column + 1 < HEIGHT - 5 && dev_cell[(cur_row + 1)*num_per_row + cur_column + 1] == 3) {
			dev_cell[(cur_row + 1)*num_per_row + cur_column + 1] = 2;
			dev_medDir[(cur_row + 1)*num_per_row + cur_column + 1] = 0;
		}
	}
	else if (dev_cell[cur_row*num_per_row + cur_column] == 3) {
		int direction = dev_medDir[cur_row*num_per_row + cur_column];
		if (direction == 1 && cur_column + 1 < HEIGHT) {
			medCellMoveCUDA(cur_row, cur_column, cur_row, cur_column + 1, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 2 && cur_row - 1 >= 5 && cur_column + 1 < HEIGHT - 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row - 1, cur_column + 1, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 3 && cur_row - 1 >= 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row - 1, cur_column, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 4 && cur_row - 1 >= 5 && cur_column - 1 >= 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row - 1, cur_column - 1, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 5 && cur_column - 1 >= 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row, cur_column - 1, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 6 && cur_row + 1 < WIDTH - 5 && cur_column - 1 >= 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row + 1, cur_column - 1, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 7 && cur_row + 1 < WIDTH - 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row + 1, cur_column, direction, dev_cell, dev_medDir, num_per_row);
		}
		else if (direction == 8 && cur_row + 1 < WIDTH - 5 && cur_column + 1 < HEIGHT - 5) {
			medCellMoveCUDA(cur_row, cur_column, cur_row + 1, cur_column + 1, direction, dev_cell, dev_medDir, num_per_row);
		}
	}
	else {
		dev_cell[cur_row*num_per_row + cur_column] = state;
	}
}

//Display individual pixels.
static void displayCUDA() {
	//bool host_ifDraw[WIDTH*HEIGHT] = { false };
	glClear(GL_COLOR_BUFFER_BIT);
	GLfloat red, green, blue;
	parallel_for(blocked_range2d<size_t>(0, WIDTH, 1000000, 0, HEIGHT, 1000000), [&](blocked_range2d<size_t> & r) {
		for (int i = r.rows().begin(); i != r.rows().end(); i++) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				if (host_cell[i*HEIGHT + j] == 1) {
					red = 1;
					green = 0;
					blue = 0;
				}
				else if (host_cell[i*HEIGHT + j] == 2) {
					red = 0;
					green = 1;
					blue = 0;
				}
				else if (host_cell[i*HEIGHT + j] == 3) {
					red = 1;
					green = 1;
					blue = 0;
				}
				else {
					red = 0.0f;
					green = 0.0f;
					blue = 0.0f;
				}
				draw(red, green, blue, i, j);
			}
		}
		glutSwapBuffers();
	});
	// declare two array variables for devices and allocate memory for them
	int * dev_cell;
	int * dev_medDir;
	//bool * dev_ifDraw;
	gpuErrchk(cudaMalloc((void**) &dev_cell, WIDTH*HEIGHT*sizeof(float)));
	gpuErrchk(cudaMalloc((void**) &dev_medDir, WIDTH*HEIGHT*sizeof(float)));

	// copy cell and medDir array from host to device
	gpuErrchk(cudaMemcpy(dev_cell, host_cell, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_medDir, host_medDir, WIDTH*HEIGHT * sizeof(float), cudaMemcpyHostToDevice));

	cellHandle<<<1536,512>>>(dev_cell, dev_medDir, 1024, 768);
	
	gpuErrchk(cudaMemcpy(host_cell, dev_cell, WIDTH*HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(host_medDir, dev_medDir, WIDTH*HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dev_cell));
	gpuErrchk(cudaFree(dev_medDir));
}

int main(int argc, char** argv)
{
	int x = 1024;
	int y = 768;
	int mult = 1;

	tbb::task_scheduler_init tbb_init;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Comp426 Assignment 3");
	init();
	setupCUDA(x, y, mult);
	glutDisplayFunc(displayCUDA);
	glutTimerFunc(0, update, 0);
	glutMouseFunc(OnMouseClick);
	glutMainLoop();
	return 0;
}
