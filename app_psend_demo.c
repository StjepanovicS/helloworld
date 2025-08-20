#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void die(const char* where, int rc)
{
  if (rc!=MPI_SUCCESS) 
  {
    char err[MPI_MAX_ERROR_STRING]; int len=0;
    MPI_Error_string(rc, err, &len);
    fprintf(stderr, "[%s] rc=%d: %.*s\n", where, rc, len, err);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  setvbuf(stdout,NULL,_IONBF,0);
  setvbuf(stderr,NULL,_IONBF,0);

  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int partitions = 1;
  const MPI_Count n = 1024;
  double *buf = (double*)malloc((size_t)partitions * n * sizeof(double));

  if (rank==0) for (size_t i=0;i<(size_t)partitions*n;i++) buf[i] = (double)i;
  else         for (size_t i=0;i<(size_t)partitions*n;i++) buf[i] = 0.0;

  MPI_Info info; int rc = MPI_Info_create(&info); die("Info_create", rc);


  // HIER BEGINNT DER PLUGIN-TEIL //
  MPI_Info_set(info, "compressor", "codec");
  MPI_Info_set(info, "compressor_plugin", "/u/home/sts/code/useful/cpu2/minimal_plugin.so");
  //MPI_Info_set(info, "codec:name", "raw");
  MPI_Info_set(info, "zfp:rate", "4");
  // Ohne Plugin: Zeilen rauskommentieren und MPI_INFO_NULL statt info

  MPI_Request req;
  int tag=0;

  if (rank==0) 
  {
    printf("[rank0] Psend_init\n");
    rc = MPI_Psend_init(buf, partitions, n, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD, info, &req); die("Psend_init", rc);

    printf("[rank0] Start\n");
    rc = MPI_Start(&req); die("Start send", rc);

    printf("[rank0] Pready_range(0,0)\n");
    rc = MPI_Pready_range(0, 0, req); die("Pready_range(0,0)", rc);

    printf("[rank0] waiting for ack\n");
    int ack = 0;
    MPI_Recv(&ack, 1, MPI_INT, 1, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("[rank0] got ack=%d\n", ack);

    printf("[rank0] Request_free\n");
    rc = MPI_Request_free(&req); die("Request_free send", rc);


  }
  
  else 
  {
    printf("[rank1] Precv_init\n");
    rc = MPI_Precv_init(buf, partitions, n, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, info, &req); die("Precv_init", rc);

    printf("[rank1] Start\n");
    rc = MPI_Start(&req); die("Start recv", rc);

    printf("[rank1] Wait\n");
    rc = MPI_Wait(&req, MPI_STATUS_IGNORE); die("Wait recv", rc);

    int ack = 1;
    MPI_Send(&ack, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
    printf("[rank1] sent ack\n");

    printf("[rank1] Request_free\n");
    rc = MPI_Request_free(&req); die("Request_free recv", rc);

    int ok=1; for (size_t i=0;i<(size_t)partitions*n;i++) if (buf[i]!=(double)i) { ok=0; break; }
    printf("[rank1] recv done, check=%s\n", ok?"OK":"FAIL");
  }

  printf("[rank%d] Barrier\n", rank);
  rc = MPI_Barrier(MPI_COMM_WORLD); die("Barrier", rc);

  free(buf);
  MPI_Info_free(&info);
  fprintf(stderr, "[rank%02d] test stderr\n", rank);
  fflush(stderr);
  MPI_Finalize();
  return 0;
}
