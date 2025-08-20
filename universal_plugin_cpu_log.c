// universal_plugin.c  —  Universal MPI compressor plugin (dispatcher)
// Supports: raw (identity), zfp (double, 1D, fixed-rate) as example codec.
// Compile as shared lib and load via MPI_Info "compressor_plugin".
// Author: you

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---- Optional codecs ----
#include <zfp.h>
#include <zfp/bitstream.h>

// #include <lz4.h> FOR LZ4


// ===== Common types / helpers =================================================

typedef enum { DT_FLOAT=1, DT_DOUBLE=2 } dtype_id_t;

static size_t dtype_size(dtype_id_t dt) {
  return (dt==DT_FLOAT)? sizeof(float) : sizeof(double);
}

#pragma pack(push,1)
typedef struct {
  uint32_t magic;      // 'D','A','R','X' ~ 0x44515258? choose fixed
  uint16_t version;    // 1
  uint16_t codec_id;   // 0=raw, 1=zfp, ...
  uint32_t flags;      // bit flags (unused now)
  uint64_t raw_bytes;  // uncompressed payload bytes
  uint32_t nx, ny, nz; // optional dims (0 if not provided)
  uint16_t dtype;      // 1=float, 2=double
  uint16_t reserved;
} codec_hdr_t;
#pragma pack(pop)

#define CODEC_MAGIC  0x58525144u /* "DQRX" arbitrary; just match both sides */
#define CODEC_VER    1

typedef struct {
  // chosen codec params
  int codec_id;              // 0 raw, 1 zfp
  dtype_id_t dtype;
  uint32_t nx, ny, nz;       // if 0 → treat as 1D of count elements
  // zfp config
  int zfp_rate_bits;         // bits/value for fixed-rate
  // per-request meta

  //int lz4_accel; FOR L4Z

  void   *base;
  int     partitions;
  MPI_Count elems_per_part;
  MPI_Datatype mpi_dtype;
  MPI_Aint max_bytes_per_part;

  // stats (optional)
  uint64_t total_deflated, total_inflated;

} req_state_t;

// ===== Codec vtable ===========================================================

typedef struct {
  const char *name;  // "raw", "zfp"
  int  id;           // 0, 1, ...
  // upper bound of compressed size for 'raw_bytes'
  size_t (*max_cbytes)(size_t raw_bytes, const req_state_t *st);
  // compress: write into out[0..out_cap), return out_size via *out_size
  int    (*compress)(const void *in, size_t raw_bytes,
                     void *out, size_t out_cap, size_t *out_size,
                     const req_state_t *st);
  // decompress: in[0..in_size) → out (raw_bytes)
  int    (*decompress)(const void *in, size_t in_size,
                       void *out, size_t raw_bytes,
                       const req_state_t *st);
} codec_vtbl_t;

// ----- RAW codec (identity) ---------------------------------------------------
static size_t raw_max_cbytes(size_t raw_bytes, const req_state_t *st) {
  (void)st; return raw_bytes;
}
static int raw_compress(const void *in, size_t raw_bytes,
                        void *out, size_t out_cap, size_t *out_size,
                        const req_state_t *st)
{
  (void)st;
  if (out_cap < raw_bytes) return MPI_ERR_TRUNCATE;
  memcpy(out, in, raw_bytes);
  *out_size = raw_bytes;
  return MPI_SUCCESS;
}
static int raw_decompress(const void *in, size_t in_size,
                          void *out, size_t raw_bytes,
                          const req_state_t *st)
{
  (void)st;
  if (in_size != raw_bytes) return MPI_ERR_TRUNCATE;
  memcpy(out, in, raw_bytes);
  return MPI_SUCCESS;
}

// ----- ZFP codec (double, 1D fixed-rate) -------------------------------------
static size_t zfp_max_cbytes(size_t raw_bytes, const req_state_t *st) {
  // raw_bytes must be N * sizeof(double) in this minimal adapter
  size_t n = raw_bytes / sizeof(double);
  zfp_type type = zfp_type_double;
  zfp_field *field = zfp_field_1d((void*)NULL, type, n);
  bitstream *stream = stream_open(NULL, 0);
  zfp_stream *z = zfp_stream_open(stream);
  zfp_stream_set_rate(z, (double)st->zfp_rate_bits, type, 1, 1);
  size_t maxsize = zfp_stream_maximum_size(z, field);
  zfp_stream_close(z); stream_close(stream); zfp_field_free(field);
  return maxsize;
}
static int zfp_do_compress_1d_double(const void *in, size_t raw_bytes,
                                     void *out, size_t out_cap, size_t *out_size,
                                     const req_state_t *st)
{
  size_t n = raw_bytes / sizeof(double);
  zfp_type type = zfp_type_double;
  zfp_field *field = zfp_field_1d((void*)in, type, n);
  bitstream *stream = stream_open(out, out_cap);
  zfp_stream *z = zfp_stream_open(stream);
  zfp_stream_set_rate(z, (double)st->zfp_rate_bits, type, 1, 1);
  size_t maxsize = zfp_stream_maximum_size(z, field);
  if (maxsize > out_cap) { zfp_stream_close(z); stream_close(stream); zfp_field_free(field); return MPI_ERR_TRUNCATE; }
  int ok = zfp_compress(z, field);
  size_t written = ok ? stream_wtell(stream) : 0; // use write tell
  zfp_stream_close(z); stream_close(stream); zfp_field_free(field);
  if (!ok) return MPI_ERR_OTHER;
  *out_size = written;
  return MPI_SUCCESS;
}
static int zfp_do_decompress_1d_double(const void *in, size_t in_size,
                                       void *out, size_t raw_bytes,
                                       const req_state_t *st)
{
  (void)st;
  size_t n = raw_bytes / sizeof(double);
  zfp_type type = zfp_type_double;
  zfp_field *field = zfp_field_1d((void*)out, type, n);
  bitstream *stream = stream_open((void*)in, in_size);
  zfp_stream *z = zfp_stream_open(stream);
  // must set the same rate (zfp uses it to configure blocks); rate comes from st
  zfp_stream_set_rate(z, (double)st->zfp_rate_bits, type, 1, 1);
  int ok = zfp_decompress(z, field);
  zfp_stream_close(z); stream_close(stream); zfp_field_free(field);
  return ok ? MPI_SUCCESS : MPI_ERR_OTHER;
}

/*
// ----- LZ4 codec (lossless, byte-oriented) -----------------------------------
static size_t lz4_max_cbytes(size_t raw_bytes, const req_state_t *st) {
  (void)st;
  return (size_t)LZ4_compressBound((int)raw_bytes);
}

static int lz4_compress_bytes(const void *in, size_t raw_bytes,
                              void *out, size_t out_cap, size_t *out_size,
                              const req_state_t *st)
{
  int cap = (int)out_cap;
  int srcSize = (int)raw_bytes;
  int written;

  if (st->lz4_accel > 1) {
    // schnellere Variante mit Acceleration (größere Zahl = schneller)
    written = LZ4_compress_fast((const char*)in, (char*)out, srcSize, cap, st->lz4_accel);
  } else {
    written = LZ4_compress_default((const char*)in, (char*)out, srcSize, cap);
  }
  if (written <= 0) return MPI_ERR_TRUNCATE;
  *out_size = (size_t)written;
  return MPI_SUCCESS;
}

static int lz4_decompress_bytes(const void *in, size_t in_size,
                                void *out, size_t raw_bytes,
                                const req_state_t *st)
{
  (void)st;
  int decoded = LZ4_decompress_safe((const char*)in, (char*)out,
                                    (int)in_size, (int)raw_bytes);
  if (decoded < 0) return MPI_ERR_OTHER;          // Dekompr.-Fehler
  if ((size_t)decoded != raw_bytes) return MPI_ERR_TRUNCATE; // Sicherheitscheck
  return MPI_SUCCESS;
}

*/

// Registry
enum
{
  CODEC_RAW=0,
  CODEC_ZFP=1
  /*, CODEC_LZ4=2*/ // Das hier für vielleicht LZ4, also lossless

};

static codec_vtbl_t g_codecs[] =
{
  { "raw", CODEC_RAW, raw_max_cbytes, raw_compress, raw_decompress },
  { "zfp", CODEC_ZFP, zfp_max_cbytes, zfp_do_compress_1d_double, zfp_do_decompress_1d_double },
  /*{ "lz4", CODEC_LZ4, lz4_max_cbytes, lz4_compress_bytes, lz4_decompress_bytes }, // <-- NEWWWWWWWWWWWWWWWW */
};

static const int g_num_codecs = sizeof(g_codecs)/sizeof(g_codecs[0]);

static const codec_vtbl_t* find_codec_by_name(const char *name, int *out_id)
{
  for (int i=0;i<g_num_codecs;i++) {
    if (strcmp(g_codecs[i].name, name)==0) { if(out_id) *out_id=g_codecs[i].id; return &g_codecs[i]; }
  }
  return NULL;
}
static const codec_vtbl_t* find_codec_by_id(int id)
{
  for (int i=0;i<g_num_codecs;i++) if (g_codecs[i].id == id) return &g_codecs[i];
  return NULL;
}

// ===== Proposal API types & registration =====================================

typedef int (MPIX_Compressor_req_init_fn)(void*, int*, MPI_Count*, MPI_Datatype*,
                                          MPI_Info, void*, MPI_Aint*, void*, void*);
typedef int (MPIX_Compressor_req_free_fn)(void*);
typedef int (MPIX_Compressor_conversion_fn)(void*, int, MPI_Count, MPI_Datatype,
                                            void*, MPI_Aint*, void*);
typedef int (MPIX_Compressor_deregister_fn)(void*);

int MPIX_Register_compressor(const char *name,
  MPIX_Compressor_req_init_fn  *req_init_fn,
  MPIX_Compressor_req_free_fn  *req_free_fn,
  MPIX_Compressor_conversion_fn *deflate_fn,
  MPIX_Compressor_conversion_fn *inflate_fn,
  MPIX_Compressor_deregister_fn *deregister_fn,
  MPI_Info compressor_info,
  void *extra_state);

// ===== Info helpers ===========================================================

static int info_get_string(MPI_Info info, const char *key, char *out, int outlen)
{
  int flag=0; MPI_Info_get_string(info, key, &outlen, out, &flag);
  return flag;
}
static int info_get_int(MPI_Info info, const char *key, int *out)
{
  char buf[64]; if (!info_get_string(info, key, buf, sizeof(buf))) return 0;
  *out = atoi(buf); return 1;
}

// ===== Callbacks ==============================================================

static int compressor_req_init(void *buf, int *partitions, MPI_Count *count, MPI_Datatype *dtype,
                               MPI_Info info, void *temp_buf, MPI_Aint *temp_buf_size,
                               void *extra_state, void *extra_req_state)
{
  (void)temp_buf; (void)extra_state;
  req_state_t *st = (req_state_t*)calloc(1, sizeof(*st));
  if (!st) return MPI_ERR_NO_MEM;

  st->base = buf;
  st->partitions = *partitions;
  st->elems_per_part = *count;
  st->mpi_dtype = *dtype;
  st->dtype = (st->mpi_dtype==MPI_FLOAT)? DT_FLOAT : DT_DOUBLE; // default only float/double

  // Defaults
  st->codec_id = CODEC_ZFP;              // default to zfp; override by info
  st->zfp_rate_bits = 4;                 // conservative
  st->nx=st->ny=st->nz=0;

  // Which codec?
  char cname[64];
  if (!info_get_string(info, "codec:name", cname, sizeof(cname))) {
    // also accept "compressor" legacy value: "codec" name -> then try "zfp" as default
    // if user set "compressor=zfp" directly, we honor that too:
    if (info_get_string(info, "compressor", cname, sizeof(cname))) {
      // allow raw/zfp here as well
    } else {
      strcpy(cname, "zfp");
    }
  }
  const codec_vtbl_t *vtbl = find_codec_by_name(cname, &st->codec_id);
  if (!vtbl) { free(st); return MPI_ERR_OTHER; }

  // Parse optional dims
  char dims[64];
  if (info_get_string(info, "codec:dimensions", dims, sizeof(dims))) {
    int nx=0,ny=0,nz=0; if (sscanf(dims, "%d,%d,%d", &nx,&ny,&nz)==3) {
      st->nx = (nx>0)? (uint32_t)nx : 0;
      st->ny = (ny>0)? (uint32_t)ny : 0;
      st->nz = (nz>0)? (uint32_t)nz : 0;
    }
  }
  // zfp config
  int rate=0;
  if (info_get_int(info, "zfp:fixed_rate", &rate) || info_get_int(info, "zfp:rate", &rate)) {
    if (rate>0) st->zfp_rate_bits = rate;
  }

  // Temp buffer size (per partition): header + codec max + small slop
  size_t elems = (size_t)st->elems_per_part;
  size_t raw_bytes = elems * dtype_size(st->dtype);
  size_t codec_cap = vtbl->max_cbytes ? vtbl->max_cbytes(raw_bytes, st) : raw_bytes;
  st->max_bytes_per_part = (MPI_Aint)(sizeof(codec_hdr_t) + codec_cap + 512);
  *temp_buf_size = st->max_bytes_per_part;

  st->total_deflated = st->total_inflated = 0;
  *(void**)extra_req_state = st;

  // *** CHANGED: stdout -> stderr ***
  fprintf(stderr, "[PLUGIN] req_init: codec=%s id=%d dtype=%s elems=%lld rate=%d cap/part=%lldB\n",
          vtbl->name, st->codec_id,
          (st->dtype==DT_FLOAT?"float":"double"),
          (long long)st->elems_per_part, st->zfp_rate_bits,
          (long long)*temp_buf_size);

  return MPI_SUCCESS;
}

static int compressor_req_free(void *extra_req_state)
{
  req_state_t *st = (req_state_t*)extra_req_state;
  if (st) {
    // *** CHANGED: stdout -> stderr ***
    fprintf(stderr, "[PLUGIN] req_free: deflated=%lluB inflated=%lluB\n",
            (unsigned long long)st->total_deflated,
            (unsigned long long)st->total_inflated);
    free(st);
  }
  return MPI_SUCCESS;
}

static int compressor_deflate(void *buf /*ign*/, int partition, MPI_Count count, MPI_Datatype dtype,
                              void *compr_buf, MPI_Aint *size, void *extra_req_state)
{
  (void)buf; (void)dtype;
  req_state_t *st = (req_state_t*)extra_req_state;
  const codec_vtbl_t* vtbl = find_codec_by_id(st->codec_id);
  if (!vtbl) return MPI_ERR_OTHER;

  size_t elems = (size_t)count;
  size_t raw_bytes = elems * dtype_size(st->dtype);
  // source partition base
  uint8_t *src = (uint8_t*)st->base + (size_t)partition * raw_bytes;

  // write header
  codec_hdr_t *hdr = (codec_hdr_t*)compr_buf;
  hdr->magic = CODEC_MAGIC;
  hdr->version = CODEC_VER;
  hdr->codec_id = (uint16_t)st->codec_id;
  hdr->flags = 0;
  hdr->raw_bytes = (uint64_t)raw_bytes;
  hdr->nx = st->nx; hdr->ny = st->ny; hdr->nz = st->nz;
  hdr->dtype = (uint16_t)st->dtype;
  hdr->reserved = 0;

  // body after header
  uint8_t *out = (uint8_t*)compr_buf + sizeof(codec_hdr_t);
  size_t out_cap = (size_t)st->max_bytes_per_part - sizeof(codec_hdr_t);
  size_t out_size = 0;

  int rc = vtbl->compress(src, raw_bytes, out, out_cap, &out_size, st);
  if (rc != MPI_SUCCESS) return rc;

  *size = (MPI_Aint)(sizeof(codec_hdr_t) + out_size);
  st->total_deflated += (uint64_t)*size;

  // *** CHANGED: stdout -> stderr ***
  fprintf(stderr, "[PLUGIN] deflate part=%d raw=%zuB -> pkt=%lldB (codec=%d)\n",
          partition, raw_bytes, (long long)*size, st->codec_id);
  return MPI_SUCCESS;
}

static int compressor_inflate(void *buf /*ign*/, int partition, MPI_Count count, MPI_Datatype dtype,
                              void *compr_buf, MPI_Aint *size, void *extra_req_state)
{
  (void)buf; (void)dtype;
  req_state_t *st = (req_state_t*)extra_req_state;

  // read header
  if ((size_t)*size < sizeof(codec_hdr_t)) return MPI_ERR_TRUNCATE;
  codec_hdr_t *hdr = (codec_hdr_t*)compr_buf;
  if (hdr->magic != CODEC_MAGIC || hdr->version != CODEC_VER) return MPI_ERR_OTHER;

  const codec_vtbl_t *vtbl = find_codec_by_id(hdr->codec_id);
  if (!vtbl) return MPI_ERR_OTHER;

  size_t raw_bytes = (size_t)hdr->raw_bytes;
  size_t elems = (size_t)count;
  size_t expect_raw = elems * dtype_size(st->dtype);
  if (expect_raw != raw_bytes) {
    // receiver view differs from sender header
    // (for this minimal plugin we require they match)
    return MPI_ERR_TRUNCATE;
  }

  // destination pointer for this partition
  uint8_t *dst = (uint8_t*)st->base + (size_t)partition * raw_bytes;

  const uint8_t *in = (const uint8_t*)compr_buf + sizeof(codec_hdr_t);
  size_t in_size = (size_t)*size - sizeof(codec_hdr_t);

  int rc = vtbl->decompress(in, in_size, dst, raw_bytes, st);
  if (rc != MPI_SUCCESS) return rc;

  st->total_inflated += (uint64_t)*size;
  // *** CHANGED: stdout -> stderr ***
  fprintf(stderr, "[PLUGIN] inflate part=%d pkt=%lldB -> raw=%zuB (codec=%d)\n",
          partition, (long long)*size, raw_bytes, hdr->codec_id);
  return MPI_SUCCESS;
}

// ===== Entry point ============================================================

__attribute__((visibility("default")))
int compressor_register(const char *compressor_name, MPI_Info info)
{
  // We register under whatever name the MPI layer passes (your app uses "codec")
  // *** CHANGED: stdout -> stderr ***
  fprintf(stderr, "[PLUGIN] register name=%s\n", compressor_name);
  return MPIX_Register_compressor(compressor_name,
                                  compressor_req_init, compressor_req_free,
                                  compressor_deflate,  compressor_inflate,
                                  /*deregister_fn*/ NULL,
                                  MPI_INFO_NULL,
                                  /*extra_state*/ NULL);
}
