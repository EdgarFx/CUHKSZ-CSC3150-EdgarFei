#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 create_time = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  // VCB size: 4096 byte = 4KB
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  // 32 bytes per FCB (20 bytes name, 4 bytes size, 2 bytes modified time, 2 bytes create time, 4 bytes file address)
  fs->FCB_SIZE = FCB_SIZE;
  // 1024 FCB (also means maxmimum 1024 files)
  fs->FCB_ENTRIES = FCB_ENTRIES;
  // 1085440 byte = 1060 KB
  fs->STORAGE_SIZE = VOLUME_SIZE;
  // 32 bytes
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  // 20 bytes
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  // maximum 1024 files
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  // maximux 1048576 byte = 1024 KB
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  // the base address to store the content of files: 4096+32768 byte = 36 KB
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // try to find the corresponding file
  int find_flag = 0;
  for(u32 i=0;i<fs->MAX_FILE_NUM;i++){
    find_flag = 1;
    for(int j=0;j<20;j++){
      if(s[j]=='\0'&&fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+j]=='\0'){
        break;
      }
      if(s[j]!=fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+j]){
        find_flag = 0;
        break;
      }
    }
    if(find_flag){
      if(op==G_READ){
        return i;
      }
      else if(op==G_WRITE){
        fs_gsys(fs,RM,s);
        break;
      }
    }
  }
  if(find_flag){
    for(u32 i=0;i<fs->MAX_FILE_NUM;i++){
      if(fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)]==0){
        // add file name
        for(int k=0;k<20;k++){
          fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+k] = s[k];
          if(s[k]=='\0'){
            break;
          }
        }
        return i;
      }
    }
  }
  else if(!find_flag){
    if(op==G_WRITE){
      // create a new file
      // find an empty FCB (filename hasn't been written)
      for(u32 i=0;i<fs->MAX_FILE_NUM;i++){
        if(fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)]==0){
          // add file name
          for(int k=0;k<20;k++){
            fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+k] = s[k];
            if(s[k]=='\0'){
              break;
            }
          }
          // add create time
          create_time++;
          *(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+6] = (uint16_t)create_time;
          return i;
        }
      }
    }
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  u32 begin_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+fp*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+8];
  for(int i=0;i<size;i++){
    output[i] = fs->volume[fs->FILE_BASE_ADDRESS+begin_block*(fs->STORAGE_BLOCK_SIZE)+i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  // update FCB
  // add size
  *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+fp*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE] = size;
  // add modified time
  gtime++;
  *(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+fp*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+4] = (uint16_t)gtime;
  // add file address, need to find an empty block, use VCB.
  for(u32 i=0;i<32768;i++){
    if((fs->volume[i/8]>>(7-i%8)&0x01)==0x00){
      *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+fp*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+8] = i;
      // write the file
      for(int j=0;j<size;j++){
        fs->volume[fs->FILE_BASE_ADDRESS+i*(fs->STORAGE_BLOCK_SIZE)+j] = input[j];
      }
      // update the VCB
      u32 bit_num;
      if(size%32==0){
        bit_num = size/32;
      }
      else{
        bit_num = size/32+1;
      }
      for(int j=i;j<i+bit_num;j++){
        fs->volume[j/8] = fs->volume[j/8]|(0x80>>(j%8));
      }
      break;
    }
  }
}


__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  if(op==LS_D){
    printf("===sort by modified time===\n");
  }
  else if(op==LS_S){
    printf("===sort by file size===\n");
  }
  int file_num=0;
  int non_empty_index[1024];
  for(int i=0;i<1024;i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)]!=0){
      non_empty_index[i] = 1;
      file_num++;
    }
    else{
      non_empty_index[i] = 0;
    }
  }
  for(int i=0;i<file_num;i++){
    int max_file_index = 0;
    for(int j=0;j<1024;j++){
      if(non_empty_index[j]!=0){
        max_file_index = j;
        break;
      }
    }   
    for(int j=0;j<1024;j++){
      if(non_empty_index[j]!=0){
        if(op==LS_D){
          if(*(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+j*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE+4]>=*(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE+4]){
            max_file_index=j;
          }
        }
        else if(op==LS_S){
          if(*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+j*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE]>*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE]){
            max_file_index=j;
          }
          else if (*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+j*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE]==*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE]){
            if(*(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+j*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE+6]<*(uint16_t*)&fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE+6]){
              max_file_index=j;
            }
          }
        }
      }
    }
    char filename[20] = {'\0'};
    for(int j=0;j<fs->MAX_FILENAME_SIZE;j++){
      if(fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*(fs->FCB_SIZE)+j]!=0){
        filename[j]=fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*(fs->FCB_SIZE)+j];
      }
    }
    if(op==LS_D){
      printf("%s\n",filename);
    }
    else if(op==LS_S){
      printf("%s %d\n",filename,*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+max_file_index*fs->FCB_SIZE+fs->MAX_FILENAME_SIZE]);
    }
    non_empty_index[max_file_index] = 0;
  }
}


__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  int find_flag = 0;
  for(u32 i=0;i<fs->MAX_FILE_NUM;i++){
    find_flag = 1;
    for(int j=0;j<20;j++){
      if(s[j]=='\0'&&fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+j]=='\0'){
        break;
      }
      if(s[j]!=fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+j]){
        find_flag = 0;
        break;
      }
    }
    if(find_flag){
      // clear FCB data
      for(int k=0;k<fs->FCB_SIZE;k++){
        fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+k] = 0;
      }
      u32 size = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE];
      u32 begin_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+i*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+8];
      // clear the file content
      for(int k=0;k<size;k++){
        fs->volume[fs->FILE_BASE_ADDRESS+begin_block*(fs->STORAGE_BLOCK_SIZE)+k] = 0;
      }
      // update the VCB
      int bit_num;
      if(size%32==0){
        bit_num = size/32;
      }
      else{
        bit_num = size/32+1;
      }
      for(int k=begin_block;k<begin_block+bit_num;k++){
        fs->volume[k/8] = fs->volume[k/8]&(uint8_t)~(0x80>>(k%8));
      }
      // implement compaction
      // check whether the deleted file is in the middle
      if(fs->volume[fs->SUPERBLOCK_SIZE+(i+1)*fs->FCB_SIZE]!=0){
        u32 next_begin_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+(i+1)*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+8];
        for(u32 k=next_begin_block;k<32768;k++){
          if((fs->volume[k/8]>>(7-k%8)&0x01)==0x00){
            break;
          }
          // compact the file content
          for(int p=0;p<fs->STORAGE_BLOCK_SIZE;p++){
            fs->volume[fs->FILE_BASE_ADDRESS+(k-(next_begin_block-begin_block))*fs->STORAGE_BLOCK_SIZE+p] = fs->volume[fs->FILE_BASE_ADDRESS+k*fs->STORAGE_BLOCK_SIZE+p];
            fs->volume[fs->FILE_BASE_ADDRESS+k*fs->STORAGE_BLOCK_SIZE+p] = 0;
          }
          // update the VCB
          fs->volume[(k-(next_begin_block-begin_block))/8] = fs->volume[(k-(next_begin_block-begin_block))/8]|(0x80>>((k-(next_begin_block-begin_block))%8));
          fs->volume[k/8] = fs->volume[k/8]&(uint8_t)~(0x80>>(k%8));
        }
        // update FCB
        for(int k=i+1;k<fs->FCB_ENTRIES;k++){
          if(fs->volume[fs->SUPERBLOCK_SIZE+k*(fs->FCB_SIZE)]!=0){
            // change the file address
            *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE+k*(fs->FCB_SIZE)+fs->MAX_FILENAME_SIZE+8] -= (next_begin_block-begin_block);
            // compact FCB
            for(int p=0;p<fs->FCB_SIZE;p++){
              fs->volume[fs->SUPERBLOCK_SIZE+(k-1)*fs->FCB_SIZE+p] = fs->volume[fs->SUPERBLOCK_SIZE+k*fs->FCB_SIZE+p];
              fs->volume[fs->SUPERBLOCK_SIZE+k*fs->FCB_SIZE+p] = 0;
            }
          }
        }
      }
      break;
    }
  }
}
