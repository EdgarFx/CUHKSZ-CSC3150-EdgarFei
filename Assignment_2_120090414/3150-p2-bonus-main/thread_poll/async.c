
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

my_queue_t *thread_pool = NULL;

static void *thread_function(void *args){
    my_queue_t *p = (my_queue_t*)args;
    my_item_t *work_item;
    while(1){
        pthread_mutex_lock(&(p->my_queue_lock));
        while(!p->head){
            pthread_cond_wait(&(p->my_queue_ready),&(p->my_queue_lock));
        }
        work_item = p->head;
        DL_DELETE(p->head,p->head);
        pthread_mutex_unlock(&(p->my_queue_lock));
        work_item->work_item_hanlder(work_item->args);
        free(work_item);
    }
    return NULL;
}


void async_init(int num_threads) {
    thread_pool = (my_queue_t*)malloc(sizeof(my_queue_t));
    pthread_mutex_init(&(thread_pool->my_queue_lock),NULL);
    pthread_cond_init(&(thread_pool->my_queue_ready),NULL);
    thread_pool->size = num_threads;
    thread_pool->thread_id = (pthread_t*)malloc(sizeof(pthread_t)*num_threads);
    thread_pool->head = NULL;
    for(int i=0;i<num_threads;i++){
        pthread_create(&(thread_pool->thread_id[i]),NULL,thread_function,thread_pool);
    }
    return;
}

void async_run(void (*hanlder)(int), int args) {
    my_item_t *work_item;
    work_item = (my_item_t*)malloc(sizeof(my_item_t));
    work_item->work_item_hanlder = hanlder;
    work_item->args = args;
    work_item->next = NULL;
    work_item->prev = NULL;

    pthread_mutex_lock(&(thread_pool->my_queue_lock));
    
    DL_APPEND(thread_pool->head,work_item);
    
    pthread_cond_signal(&(thread_pool->my_queue_ready));
    pthread_mutex_unlock(&(thread_pool->my_queue_lock));
}