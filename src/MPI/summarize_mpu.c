#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <stdlib.h>

#define send_data_tag 2001
#define return_data_tag 2002

int  softmax_mpi(int argc, char **argv, float* arr, size_t num_elements){
  int ierr;
  int my_id, root_process, num_procs, an_id;
  MPI_Status status;
   
  root_process = 0;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  printf("Hello world! I'm process %i out of %i processes\n", 
         my_id, num_procs);
  
  // 1. Master send data to slaves
  assert((num_elements % num_procs)==0);
  const int part = num_elements / num_procs;
  float global_sum = 0;
  if(my_id == root_process){
    for(int i=1; i<num_procs; ++i){
      MPI_Send(&(arr[i * part]), part, MPI_FLOAT, i, send_data_tag, MPI_COMM_WORLD);
    }
    for(int i=0; i<part; ++i){
      global_sum += arr[i];
    }
  }
  
  // 2. Slave receive data
  float* slave_arr = (float*)malloc(sizeof(float)*part);
  if(my_id != root_process){
    MPI_Recv(slave_arr, part, MPI_FLOAT,
              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("Recieve data! I'm process %i out of %i processes\n", 
          my_id, num_procs);
    float sum = 0;
    // 3. Summarize locally 
    for(int i=0; i<part; ++i){
      sum += slave_arr[i];
    }
    // 4. Send partial result to master
    MPI_Send(&sum, 1, MPI_FLOAT, root_process, send_data_tag, MPI_COMM_WORLD);
    printf("Local sum! I'm process %i out of %i processes, sum %f\n", 
          my_id, num_procs, sum);
  }
  
  
  if(my_id==root_process){
    // 5. Master receive and summarize to global result
    for(int i=1; i<num_procs; ++i){
      float tmp_sum = 0;
      MPI_Recv(&tmp_sum, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      global_sum += tmp_sum;
    }
    // 6. Send back global result
    for(int i=1; i<num_procs; ++i){
      MPI_Send(&global_sum, 1, MPI_FLOAT, i, send_data_tag, MPI_COMM_WORLD);
    }
  }else{
    // 7. Slave receive and Normalize
    MPI_Recv( &global_sum, 1, MPI_FLOAT,
              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    for(int i=0; i<part; ++i){
      slave_arr[i] = slave_arr[i] / global_sum;
    }
    // 8. Send back to master 
    MPI_Send(slave_arr, part, MPI_FLOAT, root_process, send_data_tag, MPI_COMM_WORLD);
  }

  if(my_id==root_process){
    for(int i=1; i<num_procs; ++i){
      MPI_Recv(&(arr[i * part]), part, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    // Print result
    for(int i=0; i<num_elements; ++i){
      printf("%f ", arr[i]);
    }printf("\n");
  }
  
  ierr = MPI_Finalize();
  return ierr;
}



int  softmax_mpi_scatter(int argc, char **argv, float* arr, size_t num_elements){
  int ierr;
  int my_id, root_process, num_procs, an_id;
  MPI_Status status;
   
  root_process = 0;

  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  printf("Hello world! I'm process %i out of %i processes\n", 
         my_id, num_procs);
  
  
  assert((num_elements % num_procs)==0);
  const int part = num_elements / num_procs;
  float global_sum = 0;

  float* slave_arr = (float*)malloc(sizeof(float)*part);
  // 1. Master send data to slaves
  // 2. Slave receive data
  MPI_Scatter(arr, part, MPI_FLOAT, slave_arr, part, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    
  float sum = 0;
  // 3. Summarize locally 
  for(int i=0; i<part; ++i){
    sum += slave_arr[i];
  }
  printf("Recieve data! I'm process %i out of %i processes, local sum %f\n", 
        my_id, num_procs, sum);
  // 4. Send partial result to master
  float* root_sum_buff = (float*)malloc(num_procs*sizeof(float));
  MPI_Gather(&sum, 1, MPI_FLOAT, root_sum_buff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // 5. Root compute global sum
  if(my_id==root_process){
    for(int i=0; i<num_procs; ++i){
      global_sum += root_sum_buff[i];
    }
    printf("Global sum: %f\n", global_sum);
  }

  // 6. Broadcast to other process
  MPI_Bcast(&global_sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // 7. Do normalize
  for(int i=0; i<part; ++i){
    slave_arr[i] /= global_sum;
  }

  // 8. Gather normalized result from slave
  MPI_Gather(slave_arr, part, MPI_FLOAT, arr, part, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Print result
  if(my_id==root_process){
    for(int i=0; i<num_elements; ++i){
      printf("%f ", arr[i]);
    }printf("\n");
  }
  
  ierr = MPI_Finalize();
  return ierr;
}


int main(int argc, char **argv) 
{
  float arr[] = {1, 2, 3, 4, 4, 3, 2, 1};
  // auto ierr = softmax_mpi(argc, argv, &arr[0], 8);
  auto ierr = softmax_mpi_scatter(argc, argv, &arr[0], 8);

  return ierr;
}