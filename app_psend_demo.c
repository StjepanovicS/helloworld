// app_psend_demo.c — minimal robust: 1 Partition + Request_free + Barrier + Logs
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void die(const char* where, int rc){
  if (rc!=MPI_SUCCESS) {
    char err[MPI_MAX_ERROR_STRING]; int len=0;
    MPI_Error_string(rc, err, &len);
    fprintf(stderr, "[%s] rc=%d: %.*s\n", where, rc, len, err);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  setvbuf(stdout,NULL,_IONBF,0);
  setvbuf(stderr,NULL,_IONBF,0);

  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int partitions = 1;          // wichtig: 1, um Ablauf zu entwirren
  const MPI_Count n = 1024;
  double *buf = (double*)malloc((size_t)partitions * n * sizeof(double));

  if (rank==0) for (size_t i=0;i<(size_t)partitions*n;i++) buf[i] = (double)i;
  else         for (size_t i=0;i<(size_t)partitions*n;i++) buf[i] = 0.0;

  MPI_Info info; int rc = MPI_Info_create(&info); die("Info_create", rc);
  // === wenn du Plugin testen willst: diese drei Zeilen drin lassen ===
  MPI_Info_set(info, "compressor", "codec");
  MPI_Info_set(info, "compressor_plugin", "/home/sts/code/useful/libuniversal_plugin.so");
  MPI_Info_set(info, "codec:name", "raw");   // oder "raw"
  //MPI_Info_set(info, "zfp:rate", "4");       // nur für zfpexport LD_LIBRARY_PATH=$PWD:$ZFP_HOME/lib:$LD_LIBRARY_PATH
  // === wenn du Baseline ohne Plugin testen willst: 
  //     die drei Zeilen kurz auskommentieren und später MPI_INFO_NULL statt 'info' übergeben ===

  MPI_Request req;
  int tag=0;

  if (rank==0) {
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

    /* Kein Wait/Test: direkt freigeben und weiter */
    printf("[rank0] Request_free\n");
    rc = MPI_Request_free(&req); die("Request_free send", rc);


  } else {
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

    // correctness
    int ok=1; for (size_t i=0;i<(size_t)partitions*n;i++) if (buf[i]!=(double)i) { ok=0; break; }
    printf("[rank1] recv done, check=%s\n", ok?"OK":"FAIL");
  }

  // Eine Barriere stellt sicher, dass beide Seiten „fertig“ sind,
  // bevor jemand MPI_Finalize betritt (manche Stacks mögen das lieber).
  printf("[rank%d] Barrier\n", rank);
  rc = MPI_Barrier(MPI_COMM_WORLD); die("Barrier", rc);

  free(buf);
  MPI_Info_free(&info);
  fprintf(stderr, "[rank%02d] test stderr\n", rank);
  fflush(stderr);
  MPI_Finalize();
  return 0;
}
