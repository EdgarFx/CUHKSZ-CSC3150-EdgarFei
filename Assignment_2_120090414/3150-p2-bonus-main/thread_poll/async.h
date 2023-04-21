#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>


typedef struct my_item {
  void (*work_item_hanlder)(int);
  int args;
  struct my_item *next;
  struct my_item *prev;
} my_item_t;

typedef struct my_queue {
  int size;
  my_item_t *head;
  pthread_t *thread_id;
  pthread_cond_t my_queue_ready;
  pthread_mutex_t my_queue_lock;
} my_queue_t;

void async_init(int);
void async_run(void (*fx)(int), int args);


#endif
