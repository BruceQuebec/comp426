#define GLEW_STATIC

#include "Shader.hpp"
#include "clUtils.hpp"

using namespace glm;
using namespace std;

typedef struct vPos {
	float x, y, z, w;
} vPos;

typedef struct vCols {
	float x, y, z, w;
} vCols;

// Function prototypes    
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
GLFWwindow* initOpenGL();
void initCellArray(int x, int y, int m);
void loadVboDataArray();
void initVaoVbo();
void initClContext();
void medMultiInjection();

// host parts    
const GLuint WIDTH = 1024, HEIGHT = 768;
int host_cell[WIDTH*HEIGHT] = { 0 };
int host_medDir[WIDTH*HEIGHT] = { 0 };

vPos h_vPos[WIDTH*HEIGHT] = { 0 };
vCols h_vCols[WIDTH*HEIGHT] = { 0 };
vPos h_vPos_inj_init[WIDTH*HEIGHT] = { 0 };
vCols h_vCols_inj_init[WIDTH*HEIGHT] = { 0 };
GLuint h_vertex_array[2];
GLuint h_pos_buffer[2];
GLuint h_col_buffer[2];
GLuint h_pos_buffer_inj;
GLuint h_col_buffer_inj;

/*opencl part*/
cl_context context = 0;
cl_command_queue commandQueue_GPU = 0;
cl_program program_gpu = 0;
cl_device_id devices_gpu = 0;
cl_kernel kernel_cellcheck_GPU = 0;
//cl_kernel kernel_cellcopy_GPU = 0;
cl_mem dPobj[2]; // device memory buffer for Points
cl_mem dCobj[2]; // device memory buffer for Colors
cl_mem dPobj_inj;
cl_mem dCobj_inj;

size_t GlobalWorkSize = WIDTH * HEIGHT;
size_t LocalWorkSize = 1;

bool if_injected = false;

int main()
{
	// initialize openGL context and shader
	GLFWwindow* window = initOpenGL();
	Shader shader = Shader("D:/study/concordia/2020/fall/COMP426/assignments/comp371_project/Project/source/shaders/generic.vs", "D:/study/concordia/2020/fall/COMP426/assignments/comp371_project/Project/source/shaders/generic.fs");
	
	//prepare raw data
	initCellArray(WIDTH, HEIGHT, 1);
	loadVboDataArray();
	initVaoVbo();

	//initialize openCL context
	initClContext();

	glfwSetMouseButtonCallback(window, mouse_button_callback);
	

	bool flag = true;
	int idx_display = 0;
	int idx_update = 1;
	// Game loop    
	while (!glfwWindowShouldClose(window))
	{
		//determine which buffer to dispaly or to update
		if (flag) {
			idx_display = 0;
			idx_update = 1;
			flag = false;
		}
		else {
			idx_display = 1;
			idx_update = 0;
			flag = true;
		}

		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions    
		glfwPollEvents();
		
		// Render    
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		shader.use();
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(-1.0f, -1.0f, 0.0f));
		model = glm::scale(model, glm::vec3(0.005f, 0.005f, 0.005f));
		shader.setMat4("model", model);
		
		glBindVertexArray(h_vertex_array[idx_display]);
		glDrawArrays(GL_POINTS, 0, WIDTH*HEIGHT);

		clSetKernelArg(kernel_cellcheck_GPU, 0, sizeof(cl_mem), &dPobj[idx_update]);
		clSetKernelArg(kernel_cellcheck_GPU, 1, sizeof(cl_mem), &dCobj[idx_update]);
		clSetKernelArg(kernel_cellcheck_GPU, 2, sizeof(cl_mem), &dPobj_inj);
		clSetKernelArg(kernel_cellcheck_GPU, 3, sizeof(cl_mem), &dCobj_inj);
		// acquire the vertex buffers from opengl:
		cl_int status;
		status = clEnqueueAcquireGLObjects(commandQueue_GPU, 1, &dPobj[idx_update], 0, NULL, NULL);
		status = clEnqueueAcquireGLObjects(commandQueue_GPU, 1, &dCobj[idx_update], 0, NULL, NULL);
		status = clEnqueueAcquireGLObjects(commandQueue_GPU, 1, &dPobj_inj, 0, NULL, NULL);
		status = clEnqueueAcquireGLObjects(commandQueue_GPU, 1, &dCobj_inj, 0, NULL, NULL);
		
		// enqueue the Kernel object for execution:
		cl_event wait;
		status = clEnqueueNDRangeKernel(commandQueue_GPU, kernel_cellcheck_GPU, 1, NULL, &GlobalWorkSize, &LocalWorkSize, 0, NULL, &wait);
		clEnqueueReleaseGLObjects(commandQueue_GPU, 1, &dCobj[idx_update], 0, NULL, NULL);
		clEnqueueReleaseGLObjects(commandQueue_GPU, 1, &dPobj[idx_update], 0, NULL, NULL);
		clEnqueueReleaseGLObjects(commandQueue_GPU, 1, &dPobj_inj, 0, NULL, NULL);
		clEnqueueReleaseGLObjects(commandQueue_GPU, 1, &dCobj_inj, 0, NULL, NULL);
		
		glBindBuffer(GL_COPY_READ_BUFFER, h_pos_buffer[idx_update]);
		glBindBuffer(GL_COPY_WRITE_BUFFER, h_pos_buffer[idx_display]);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(h_vPos));

		glBindBuffer(GL_COPY_READ_BUFFER, h_col_buffer[idx_update]);
		glBindBuffer(GL_COPY_WRITE_BUFFER, h_col_buffer[idx_display]);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(h_vCols));

		// Swap the screen buffers    
		glfwSwapBuffers(window);
		glFlush();
	}
	// Terminate GLFW, clearing any resources allocated by GLFW.    
	glfwTerminate();
	
	clReleaseContext(context);
	clReleaseKernel(kernel_cellcheck_GPU);
	clReleaseProgram(program_gpu);
	clReleaseCommandQueue(commandQueue_GPU);
	clReleaseMemObject(dPobj[0]);
	clReleaseMemObject(dCobj[0]);
	clReleaseMemObject(dPobj[1]);
	clReleaseMemObject(dCobj[1]);
	clReleaseMemObject(dPobj_inj);
	clReleaseMemObject(dCobj_inj);
	exit(0);
}

/*openGL part*/
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	std::cout << key << std::endl;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		/*GLdouble xpos, ypos;
		//getting cursor position
		glfwGetCursorPos(window, &xpos, &ypos);
		GLdouble worldX = xpos;
		GLdouble worldY = ypos;*/

		medMultiInjection();
	}
}

GLFWwindow* initOpenGL() {
	std::cout << "Starting GLFW context, OpenGL 3.3" << std::endl;
	// Init GLFW    
	glfwInit();
	// Set all the required options for GLFW    
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	// Create a GLFWwindow object that we can use for GLFW's functions    
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);

	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);
	// Set the required callback functions    
	glfwSetKeyCallback(window, key_callback);
	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions    
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers    
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
	}
	glDisable(GL_DEPTH_TEST);
	// Define the viewport dimensions    
	glViewport(0, 0, WIDTH, HEIGHT);
	glPointSize(3.0);
	return window;
}

/*project function part*/
/*initialize host cell array*/
void initCellArray(int x, int y, int m) {
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

/*initialize vertex buffer object array*/
void loadVboDataArray() {
	for (int i = 0; i < WIDTH*HEIGHT; ++i) {
		int cur_row = (int) i / WIDTH;
		int cur_column = i % WIDTH;
		
		h_vPos[i].x = cur_row;
		h_vPos[i].y = cur_column;
		h_vPos[i].z = host_medDir[i];
		h_vPos[i].w = host_cell[i];

		h_vPos_inj_init[i].x = 0;
		h_vPos_inj_init[i].y = 0;
		h_vPos_inj_init[i].z = 0;
		h_vPos_inj_init[i].w = 0;

		h_vCols_inj_init[i].x = 0;
		h_vCols_inj_init[i].y = 0;
		h_vCols_inj_init[i].z = 0;
		h_vCols_inj_init[i].w = 0;

		if ((int) h_vPos[i].w == 1) {
			h_vCols[i].x = 1;
			h_vCols[i].y = 0;
			h_vCols[i].z = 0;
			h_vCols[i].w = 1;
		}
		else if ((int) h_vPos[i].w == 2) {
			h_vCols[i].x = 0;
			h_vCols[i].y = 1;
			h_vCols[i].z = 0;
			h_vCols[i].w = 1;
		}
		else {
			h_vCols[i].x = 1;
			h_vCols[i].y = 1;
			h_vCols[i].z = 0;
			h_vCols[i].w = 1;
		}
	}
}

void initVaoVbo() {
	glGenVertexArrays(1, &h_vertex_array[0]);
	glBindVertexArray(h_vertex_array[0]);
	glGenBuffers(1, &h_pos_buffer[0]);
	glBindBuffer(GL_ARRAY_BUFFER, h_pos_buffer[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vPos), h_vPos, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GL_FLOAT), (void *)0);
	glGenBuffers(1, &h_col_buffer[0]);
	glBindBuffer(GL_ARRAY_BUFFER, h_col_buffer[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vCols), h_vCols, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GL_FLOAT), (void *)0);


	glGenVertexArrays(1, &h_vertex_array[1]);
	glBindVertexArray(h_vertex_array[1]);
	glGenBuffers(1, &h_pos_buffer[1]);
	glBindBuffer(GL_ARRAY_BUFFER, h_pos_buffer[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vPos), h_vPos, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), (void *)0);
	glGenBuffers(1, &h_col_buffer[1]);
	glBindBuffer(GL_ARRAY_BUFFER, h_col_buffer[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vCols), h_vCols, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), (void *)0);

	glGenBuffers(1, &h_pos_buffer_inj);
	glBindBuffer(GL_ARRAY_BUFFER, h_pos_buffer_inj);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vPos_inj_init), NULL, GL_STATIC_DRAW);

	glGenBuffers(1, &h_col_buffer_inj);
	glBindBuffer(GL_ARRAY_BUFFER, h_col_buffer_inj);
	glBufferData(GL_ARRAY_BUFFER, sizeof(h_vCols_inj_init), NULL, GL_STATIC_DRAW);
}


void initClContext() {
	int err;
	context = CreateContext(devices_gpu);
	commandQueue_GPU = CreateCommandQueue(context, &devices_gpu, CL_DEVICE_TYPE_GPU);
	program_gpu = CreateProgram(context, devices_gpu, "D:\\study\\concordia\\2020\\fall\\COMP426\\assignments\\comp371_project\\Project\\source\\cellStateHandle.cl", "");
	kernel_cellcheck_GPU = clCreateKernel(program_gpu, "cellHandle", NULL);
	dPobj[0] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_pos_buffer[0], &err);
	dCobj[0] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_col_buffer[0], &err);
	dPobj[1] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_pos_buffer[1], &err);
	dCobj[1] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_col_buffer[1], &err);
	dPobj_inj = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_pos_buffer_inj, &err);
	dCobj_inj = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, h_col_buffer_inj, &err);
}

void medMultiInjection() {

	//variable indicating how many groups of medicine should be injected at a time
	int num_medicines_group = rand() % (15) + 1;
	int inject_area_width = 800;
	int inject_area_height = 600;
	int h = (1 * inject_area_width) + 2;
	int w = (1 * inject_area_height) + 2;
	bool ifInjected[WIDTH*HEIGHT] = { false };
	
	for (int i = 0; i < num_medicines_group; ++i) {
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
		
		glBindBuffer(GL_ARRAY_BUFFER, h_pos_buffer_inj);
		struct vPos* h_vPos_inj = (struct vPos *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
		
		for (int j = center_x - (int)num_medicine_factor / 2; j < center_x + (int)num_medicine_factor / 2 + 1; ++j) {
			//printf("num_medicine_factor is %d, num_medicine is %d, current x is %d\n", num_medicine_factor, num_medicine, j);
			h_vPos_inj[j*HEIGHT + center_y + num_medicine_factor].x = j;
			h_vPos_inj[j*HEIGHT + center_y + num_medicine_factor].y = (center_y + num_medicine_factor);
			h_vPos_inj[j*HEIGHT + center_y + num_medicine_factor].z = 1;
			h_vPos_inj[j*HEIGHT + center_y + num_medicine_factor].w = 3;
			
			h_vPos_inj[j*HEIGHT + center_y - num_medicine_factor].x = j;
			h_vPos_inj[j*HEIGHT + center_y - num_medicine_factor].y = (center_y - num_medicine_factor);
			h_vPos_inj[j*HEIGHT + center_y - num_medicine_factor].z = 5;
			h_vPos_inj[j*HEIGHT + center_y - num_medicine_factor].w = 3;
		}

		for (int j = center_y - (int)num_medicine_factor / 2; j < center_y + (int)num_medicine_factor / 2 + 1; ++j) {
			h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + j].x = (center_x - num_medicine_factor);
			h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + j].y = j;
			h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + j].z = 3;
			h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + j].w = 3;

			h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + j].x = (center_x + num_medicine_factor);
			h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + j].y = j;
			h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + j].z = 7;
			h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + j].w = 3;
		}

		//left_up
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].x = (center_x - num_medicine_factor);
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].y = center_y + num_medicine_factor;
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].z = 2;
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].w = 3;

		//left_down
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].x = (center_x - num_medicine_factor);
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].y = center_y - num_medicine_factor;
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].z = 4;
		h_vPos_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].w = 3;

		//right_down
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].x = (center_x + num_medicine_factor);
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].y = center_y - num_medicine_factor;
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].z = 6;
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].w = 3;

		//right_up
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].x = (center_x + num_medicine_factor);
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].y = center_y + num_medicine_factor;
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].z = 8;
		h_vPos_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].w = 3;

		glUnmapBuffer(GL_ARRAY_BUFFER);

		glBindBuffer(GL_ARRAY_BUFFER, h_col_buffer_inj);
		struct vCols *h_vCols_inj = (struct vCols *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
		for (int j = center_x - (int)num_medicine_factor / 2; j < center_x + (int)num_medicine_factor / 2 + 1; ++j) {
			//printf("num_medicine_factor is %d, num_medicine is %d, current x is %d\n", num_medicine_factor, num_medicine, j);
			h_vCols_inj[j*HEIGHT + center_y + num_medicine_factor].x = 1;
			h_vCols_inj[j*HEIGHT + center_y + num_medicine_factor].y = 1;
			h_vCols_inj[j*HEIGHT + center_y + num_medicine_factor].z = 0;
			h_vCols_inj[j*HEIGHT + center_y + num_medicine_factor].w = 100;

			h_vCols_inj[j*HEIGHT + center_y - num_medicine_factor].x = 1;
			h_vCols_inj[j*HEIGHT + center_y - num_medicine_factor].y = 1;
			h_vCols_inj[j*HEIGHT + center_y - num_medicine_factor].z = 0;
			h_vCols_inj[j*HEIGHT + center_y - num_medicine_factor].w = 100;
		}

		for (int j = center_y - (int)num_medicine_factor / 2; j < center_y + (int)num_medicine_factor / 2 + 1; ++j) {
			h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + j].x = 1;
			h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + j].y = 1;
			h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + j].z = 0;
			h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + j].w = 100;

			h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + j].x = 1;
			h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + j].y = 1;
			h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + j].z = 0;
			h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + j].w = 100;
		}

		//left_up
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].x = 1;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].y = 1;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].z = 0;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].w = 100;

		//left_down
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].x = 1;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].y = 1;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].z = 0;
		h_vCols_inj[(center_x - num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].w = 100;

		//right_down
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].x = 1;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].y = 1;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].z = 0;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y - num_medicine_factor].w = 100;

		//right_up
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].x = 1;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].y = 1;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].z = 0;
		h_vCols_inj[(center_x + num_medicine_factor)*HEIGHT + center_y + num_medicine_factor].w = 100;
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}
}