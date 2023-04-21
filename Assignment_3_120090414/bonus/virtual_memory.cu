#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    vm->invert_page_table[i + 2*(vm->PAGE_ENTRIES)] = 0; // The array used to implement LRU
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  u32 VPN = addr >> 5;
  u32 offset = addr & 0x1f;

  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x00000000){//the page is valid, frame not empty
      if(vm->invert_page_table[i+vm->PAGE_ENTRIES]==VPN && vm->invert_page_table[i+3*(vm->PAGE_ENTRIES)]==threadIdx.x){ //page can be found in the page table
        for(int j=0;j<vm->PAGE_ENTRIES;j++){
          if(vm->invert_page_table[j]==0x00000000){
            if(j==i){
              vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] = 0;
            }
            else{
              vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] += 1;
            }
          }
        }
        return vm->buffer[i*(vm->PAGESIZE)+offset];
      }
    }
  }
  // if now haven't return, then there is a page fault
  *(vm->pagefault_num_ptr) += 1;
  // consider whether we have empty frame
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x80000000){ //invalid
      vm->invert_page_table[i] = 0x00000000;
      vm->invert_page_table[i+vm->PAGE_ENTRIES] = VPN;
      vm->invert_page_table[i+3*(vm->PAGE_ENTRIES)] = threadIdx.x;
      for(int j=0;j<vm->PAGE_ENTRIES;j++){
        if(vm->invert_page_table[j]==0x00000000){
          if(j==i){
            vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] = 0;
          }
          else{
            vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] += 1;
          }
        }
      }
      // load the data from storage to main memory
      for(int j=0;j<vm->PAGESIZE;j++){
        vm->buffer[i*(vm->PAGESIZE)+j] = vm->storage[VPN*(vm->PAGESIZE)+j];
      }
      return vm->buffer[i*(vm->PAGESIZE)+offset];
    }
  }
  // otherwise, the page table is full, we need to do the swap operation using LRU
  int LR_num=0;
  int LR_entry;
  // find the least recently used entry
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)]>LR_num){
      LR_num = vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)];
      LR_entry = i;
    }
  }
  // swap the data
  for(int i=0;i<vm->PAGESIZE;i++){
    vm->storage[vm->invert_page_table[LR_entry+vm->PAGE_ENTRIES]*(vm->PAGESIZE)+i] = vm->buffer[LR_entry*(vm->PAGESIZE)+i];
    vm->buffer[LR_entry*(vm->PAGESIZE)+i] = vm->storage[VPN*(vm->PAGESIZE)+i];
  }
  // update the page table
  vm->invert_page_table[LR_entry+vm->PAGE_ENTRIES] = VPN;
  vm->invert_page_table[LR_entry+3*(vm->PAGE_ENTRIES)] = threadIdx.x;
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x00000000){
      if(i==LR_entry){
        vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)] = 0;
      }
      else{
        vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)] += 1;
      }
    }
  }
  return vm->buffer[LR_entry*vm->PAGESIZE+offset];
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 VPN = addr >> 5;
  u32 offset = addr & 0x1f;

  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x00000000){
      if(vm->invert_page_table[i+vm->PAGE_ENTRIES]==VPN && vm->invert_page_table[i+3*(vm->PAGE_ENTRIES)]==threadIdx.x){
        for(int j=0;j<vm->PAGE_ENTRIES;j++){
          if(vm->invert_page_table[j]==0x00000000){
            if(j==i){
              vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] = 0;
            }
            else{
              vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] += 1;
            }
          }
        }
        vm->buffer[i*(vm->PAGESIZE)+offset] = value;
        return;
      }
    }
  }
  // if now haven't return, then there is a page fault
  *(vm->pagefault_num_ptr) += 1;
  // consider whether we have empty frame
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x80000000){ //invalid
      vm->invert_page_table[i] = 0x00000000;
      vm->invert_page_table[i+vm->PAGE_ENTRIES] = VPN;
      vm->invert_page_table[i+3*(vm->PAGE_ENTRIES)] = threadIdx.x;
      for(int j=0;j<vm->PAGE_ENTRIES;j++){
        if(vm->invert_page_table[j]==0x00000000){
          if(j==i){
            vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] = 0;
          }
          else{
            vm->invert_page_table[j+2*(vm->PAGE_ENTRIES)] += 1;
          }
        }
      }
      // load the data from storage to main memory
      vm->buffer[i*(vm->PAGESIZE)+offset] = value;
      return;
    }
  }
  // otherwise, the page table is full, we need to do the swap operation using LRU
  int LR_num=0;
  int LR_entry;
  // find the least recently used entry
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)]>LR_num){
      LR_num = vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)];
      LR_entry = i;
    }
  }
  // swap the data
  for(int i=0;i<vm->PAGESIZE;i++){
    vm->storage[vm->invert_page_table[LR_entry+vm->PAGE_ENTRIES]*(vm->PAGESIZE)+i] = vm->buffer[LR_entry*(vm->PAGESIZE)+i];
  }
  // update the page table
  vm->invert_page_table[LR_entry+vm->PAGE_ENTRIES] = VPN;
  vm->invert_page_table[LR_entry+3*(vm->PAGE_ENTRIES)] = threadIdx.x;
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(vm->invert_page_table[i]==0x00000000){
      if(i==LR_entry){
        vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)] = 0;
      }
      else{
        vm->invert_page_table[i+2*(vm->PAGE_ENTRIES)] += 1;
      }
    }
  }
  vm->buffer[LR_entry*vm->PAGESIZE+offset] = value;
  return;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i=0;i<input_size;i++){
    results[i] = vm_read(vm,i+offset);
  }
}

