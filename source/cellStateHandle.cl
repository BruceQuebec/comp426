typedef float4 Point;
typedef float4 Color;

void checkStatus(__global Point* dPobj, int cur_row, int cur_column, int* state) {
	int cancerNeighbours = 0;
	int liveNeighbours = 0;
	int medNeighbours = 0;
	//int cur_row = (int)dPobj[curIdx].x;
	//int cur_column = (int)dPobj[curIdx].y;
	int num_per_row = 768;

	int upper_row = cur_row - 1 < 0 ? 0 : cur_row - 1;
	int lower_row = cur_row + 1 > 1023 ? 1023 : cur_row + 1;
	int left_column = cur_column - 1 < 0 ? 0 : cur_column - 1;
	int right_column = cur_column + 1 > 767 ? 767 : cur_column + 1;
	for (int i = upper_row; i <= lower_row; i++) {

		if ((int) dPobj[i*num_per_row + left_column].w == 1 && left_column != cur_column) {
			cancerNeighbours++;
		}
		else if ((int)dPobj[i*num_per_row + left_column].w == 2 && left_column != cur_column) {
			liveNeighbours++;
		}
		else if ((int)dPobj[i*num_per_row + left_column].w == 3 && left_column != cur_column) {
			medNeighbours++;
		}
		if ((int)dPobj[i*num_per_row + right_column].w == 1 && right_column != cur_column) {
			cancerNeighbours++;
		}
		else if ((int)dPobj[i*num_per_row + right_column].w == 2 && right_column != cur_column) {
			liveNeighbours++;
		}
		else if ((int)dPobj[i*num_per_row + right_column].w == 3 && right_column != cur_column) {
			medNeighbours++;
		}
	}
	if ((int)dPobj[cur_row*num_per_row + cur_column].w != 0) {
		if ((int)dPobj[upper_row*num_per_row + cur_column].w == 1 && upper_row != cur_row) {
			cancerNeighbours++;
		}
		else if ((int)dPobj[upper_row*num_per_row + cur_column].w == 2 && upper_row != cur_row) {
			liveNeighbours++;
		}
		else if ((int)dPobj[upper_row*num_per_row + cur_column].w == 3 && upper_row != cur_row) {
			medNeighbours++;
		}

		if ((int)dPobj[lower_row*num_per_row + cur_column].w == 1 && lower_row != cur_row) {
			cancerNeighbours++;
		}
		else if ((int)dPobj[lower_row*num_per_row + cur_column].w == 2 && lower_row != cur_row) {
			liveNeighbours++;
		}
		else if ((int)dPobj[lower_row*num_per_row + cur_column].w == 3 && lower_row != cur_row) {
			medNeighbours++;
		}

		if ((int)dPobj[cur_row*num_per_row + cur_column].w == 1 && medNeighbours >= 3) {
			*state = 2;
		}
		else if ((int)dPobj[cur_row*num_per_row + cur_column].w == 2 && cancerNeighbours >= 5) {
			*state = 1;
		}
		else
			*state = (int)dPobj[cur_row*num_per_row + cur_column].w;
	}
	else {
		if (liveNeighbours >= 2) {
			*state = 2;
		}
	}
}

void medCellMove(int o_x, int o_y, int t_x, int t_y, float direction, __global Point* dPobj, __global Color* dCobj) {
	
	int num_per_row = 768;
	int typeTemp = (int) dPobj[t_x * num_per_row + t_y].w;
	//printf("the direction is: %d\n", (int) direction);
	
	dPobj[o_x * num_per_row + o_y].w = typeTemp;
	dPobj[t_x * num_per_row + t_y].w = 3;
	dPobj[o_x * num_per_row + o_y].z = 0;
	dPobj[t_x * num_per_row + t_y].z = direction;

	Color colorTemp = dCobj[t_x * num_per_row + t_y];
	dCobj[t_x * num_per_row + t_y].x = dCobj[o_x * num_per_row + o_y].x;
	dCobj[t_x * num_per_row + t_y].y = dCobj[o_x * num_per_row + o_y].y;
	dCobj[t_x * num_per_row + t_y].z = dCobj[o_x * num_per_row + o_y].z;
	dCobj[t_x * num_per_row + t_y].w = 999;
	dCobj[o_x * num_per_row + o_y].x = colorTemp.x;
	dCobj[o_x * num_per_row + o_y].y = colorTemp.y;
	dCobj[o_x * num_per_row + o_y].z = colorTemp.z;
	dCobj[o_x * num_per_row + o_y].w = 999;
}

__kernel void cellHandle(__global Point* dPobj, __global Color* dCobj, __global Point* dPobj_inj, __global Color* dCobj_inj) {
	int idx = get_global_id(0);

	int WIDTH = 1024;
	int HEIGHT = 768;
	int num_per_row = HEIGHT;

	int cur_row = idx*4 / sizeof(float) / num_per_row;
	int cur_column = idx*4 / sizeof(float) % num_per_row;

	int state;
	
	if ((int)dCobj_inj[idx].w == 100 && (int)dPobj[idx].w != 3) {
		
		barrier(CLK_GLOBAL_MEM_FENCE);
		dPobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].z = dPobj_inj[idx].z;
		dPobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].w = dPobj_inj[idx].w;

		dCobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].x = 1;
		dCobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].y = 1;
		dCobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].z = 0;
		dCobj[(int)dPobj_inj[idx].x*num_per_row + (int)dPobj_inj[idx].y].w = 999;

		dPobj_inj[idx].x = 0;
		dPobj_inj[idx].y = 0;
		dPobj_inj[idx].z = 0;
		dPobj_inj[idx].w = 0;
		dCobj_inj[idx].x = 0;
		dCobj_inj[idx].y = 0;
		dCobj_inj[idx].z = 0;
		dCobj_inj[idx].w = 0;
	}
	
	
	
	if (dCobj[idx].w == 999) {
		dCobj[idx].w = 1;
	}
	else {
		checkStatus(dPobj, cur_row, cur_column, &state);
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (state == 2 && (int)dPobj[idx].w == 1) {
			if (cur_column + 1 < HEIGHT - 5 && (int)dPobj[cur_row*num_per_row + cur_column + 1].w == 3) {
				//printf("the color of healthy cell is: %d %d %d\n", (int)dCobj[cur_row*num_per_row + cur_column + 1].x, (int)dCobj[cur_row*num_per_row + cur_column + 1].y, (int)dCobj[cur_row*num_per_row + cur_column + 1].z);
				dPobj[cur_row*num_per_row + cur_column + 1].w = 2;
				dPobj[cur_row*num_per_row + cur_column + 1].z = 0;

				dCobj[cur_row*num_per_row + cur_column + 1].x = 0;
				dCobj[cur_row*num_per_row + cur_column + 1].y = 1;
				dCobj[cur_row*num_per_row + cur_column + 1].z = 0;
				dCobj[cur_row*num_per_row + cur_column + 1].w = 999;
				//printf("new color x is: %d, y is %d, z is %d\n", (int)dCobj[cur_row*num_per_row + cur_column + 1].x, (int)dCobj[cur_row*num_per_row + cur_column + 1].y, (int)dCobj[cur_row*num_per_row + cur_column + 1].z);
			}
			if (cur_row - 1 >= 5 && cur_column + 1 < HEIGHT - 5 && (int)dPobj[(cur_row - 1)*num_per_row + cur_column + 1].w == 3) {
				dPobj[(cur_row - 1)*num_per_row + cur_column + 1].w = 2;
				dPobj[(cur_row - 1)*num_per_row + cur_column + 1].z = 0;

				dCobj[(cur_row - 1)*num_per_row + cur_column + 1].x = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column + 1].y = 1;
				dCobj[(cur_row - 1)*num_per_row + cur_column + 1].z = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column + 1].w = 999;
			}
			if (cur_row - 1 >= 5 && (int)dPobj[(cur_row - 1)*num_per_row + cur_column].w == 3) {
				dPobj[(cur_row - 1)*num_per_row + cur_column].w = 2;
				dPobj[(cur_row - 1)*num_per_row + cur_column].z = 0;

				dCobj[(cur_row - 1)*num_per_row + cur_column].x = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column].y = 1;
				dCobj[(cur_row - 1)*num_per_row + cur_column].z = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column].w = 999;
			}
			if (cur_row - 1 >= 5 && cur_column - 1 >= 5 && (int)dPobj[(cur_row - 1)*num_per_row + cur_column - 1].w == 3) {
				dPobj[(cur_row - 1)*num_per_row + cur_column - 1].w = 2;
				dPobj[(cur_row - 1)*num_per_row + cur_column - 1].z = 0;

				dCobj[(cur_row - 1)*num_per_row + cur_column - 1].x = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column - 1].y = 1;
				dCobj[(cur_row - 1)*num_per_row + cur_column - 1].z = 0;
				dCobj[(cur_row - 1)*num_per_row + cur_column - 1].w = 999;
			}
			if (cur_column - 1 >= 5 && (int)dPobj[cur_row*num_per_row + cur_column - 1].w == 3) {
				dPobj[cur_row*num_per_row + cur_column - 1].w = 2;
				dPobj[cur_row*num_per_row + cur_column - 1].z = 0;

				dCobj[cur_row*num_per_row + cur_column - 1].x = 0;
				dCobj[cur_row*num_per_row + cur_column - 1].y = 1;
				dCobj[cur_row*num_per_row + cur_column - 1].z = 0;
				dCobj[cur_row*num_per_row + cur_column - 1].w = 999;
			}
			if (cur_row + 1 < WIDTH - 5 && cur_column - 1 >= 5 && (int)dPobj[(cur_row + 1)*num_per_row + cur_column - 1].w == 3) {
				dPobj[(cur_row + 1)*num_per_row + cur_column - 1].w = 2;
				dPobj[(cur_row + 1)*num_per_row + cur_column - 1].z = 0;

				dCobj[(cur_row + 1)*num_per_row + cur_column - 1].x = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column - 1].y = 1;
				dCobj[(cur_row + 1)*num_per_row + cur_column - 1].z = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column - 1].w = 999;
			}
			if (cur_row + 1 < WIDTH - 5 && (int)dPobj[(cur_row + 1)*num_per_row + cur_column].w == 3) {
				dPobj[(cur_row + 1)*num_per_row + cur_column].w = 2;
				dPobj[(cur_row + 1)*num_per_row + cur_column].z = 0;

				dCobj[(cur_row + 1)*num_per_row + cur_column].x = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column].y = 1;
				dCobj[(cur_row + 1)*num_per_row + cur_column].z = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column].w = 999;
			}
			if (cur_row + 1 < WIDTH - 5 && cur_column + 1 < HEIGHT - 5 && (int)dPobj[(cur_row + 1)*num_per_row + cur_column + 1].w == 3) {
				dPobj[(cur_row + 1)*num_per_row + cur_column + 1].w = 2;
				dPobj[(cur_row + 1)*num_per_row + cur_column + 1].z = 0;

				dCobj[(cur_row + 1)*num_per_row + cur_column + 1].x = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column + 1].y = 1;
				dCobj[(cur_row + 1)*num_per_row + cur_column + 1].z = 0;
				dCobj[(cur_row + 1)*num_per_row + cur_column + 1].w = 999;
			}
		}
		else if ((int)dPobj[idx].w == 3) {
			
			int direction = (int)dPobj[idx].z;
			if (direction == 1 && cur_column + 1 < HEIGHT - 5) {
				medCellMove(cur_row, cur_column, cur_row, cur_column + 1, direction, dPobj, dCobj);
			}
			else if (direction == 2 && cur_row - 1 >= 5 && cur_column + 1 < HEIGHT - 5) {
				medCellMove(cur_row, cur_column, cur_row - 1, cur_column + 1, direction, dPobj, dCobj);
			}
			else if (direction == 3 && cur_row - 1 >= 5) {
				medCellMove(cur_row, cur_column, cur_row - 1, cur_column, direction, dPobj, dCobj);
			}
			else if (direction == 4 && cur_row - 1 >= 5 && cur_column - 1 >= 5) {
				medCellMove(cur_row, cur_column, cur_row - 1, cur_column - 1, direction, dPobj, dCobj);
			}
			else if (direction == 5 && cur_column - 1 >= 5) {
				medCellMove(cur_row, cur_column, cur_row, cur_column - 1, direction, dPobj, dCobj);
			}
			else if (direction == 6 && cur_row + 1 < WIDTH - 5 && cur_column - 1 >= 5) {
				medCellMove(cur_row, cur_column, cur_row + 1, cur_column - 1, direction, dPobj, dCobj);
			}
			else if (direction == 7 && cur_row + 1 < WIDTH - 5) {
				medCellMove(cur_row, cur_column, cur_row + 1, cur_column, direction, dPobj, dCobj);
			}
			else if (direction == 8 && cur_row + 1 < WIDTH - 5 && cur_column + 1 < HEIGHT - 5) {
				medCellMove(cur_row, cur_column, cur_row + 1, cur_column + 1, direction, dPobj, dCobj);
			}
		}
		else {
			dPobj[idx].w = state;
			if (state == 1) {
				dCobj[idx].x = 1;
				dCobj[idx].y = 0;
				dCobj[idx].z = 0;
			}
			else {
				dCobj[idx].x = 0;
				dCobj[idx].y = 1;
				dCobj[idx].z = 0;
			}
		}
	}
	
}