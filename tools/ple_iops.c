// Random 4 KiB O_DIRECT read IOPS of the PLE table's drive by thread count (single-stream lever 4).
//   gcc -O2 -pthread -o /tmp/edge0-iops tools/ple_iops.c && /tmp/edge0-iops <ple-fp8.gguf>
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static int fd;
static long long fsize;
static int per_thread;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *work(void *a) {
  unsigned s = (unsigned)(uintptr_t)a * 2654435761u + (unsigned)time(0);
  void *buf;
  if (posix_memalign(&buf, 4096, 4096)) return 0;
  for (int i = 0; i < per_thread; i++) {
    unsigned long long r = ((unsigned long long)rand_r(&s) << 31) ^ (unsigned long long)rand_r(&s);
    long long off = (long long)(r % (unsigned long long)(fsize / 4096)) * 4096;
    if (pread(fd, buf, 4096, off) != 4096) {
      perror("pread");
      exit(1);
    }
  }
  free(buf);
  return 0;
}

int main(int c, char **v) {
  if (c < 2) return 1;
  fd = open(v[1], O_RDONLY | O_DIRECT);
  fsize = lseek(fd, 0, SEEK_END);
  int qs[] = {1, 4, 16, 32, 64};
  for (int q = 0; q < 5; q++) {
    int n = qs[q];
    per_thread = 2048 / n;
    pthread_t th[64];
    double t0 = now();
    for (int i = 0; i < n; i++) pthread_create(&th[i], 0, work, (void *)(uintptr_t)(i + 1));
    for (int i = 0; i < n; i++) pthread_join(th[i], 0);
    double dt = now() - t0;
    printf("threads %2d: %d reads in %.1f ms = %.0f IOPS, %.1f us per read amortized\n", n, per_thread * n,
           dt * 1e3, per_thread * n / dt, dt * 1e6 / (per_thread * n));
  }
  return 0;
}
