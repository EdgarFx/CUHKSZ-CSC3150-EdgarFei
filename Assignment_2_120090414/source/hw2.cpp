#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 

pthread_mutex_t mutex;

int flag = 0; //flag=0:normal; flag=1:win; flag=-1:lose; flag=2:quit
int usleep_time = 100000;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move(void *t){
	/* Initialize the logs */
	int log_start[ROW-1];
	for(int i=1;i<ROW;i++){
		log_start[i-1] = rand()%(COLUMN-1);
		for(int j=0;j<15;j++){
			map[i][(log_start[i-1]+j)%(COLUMN-1)] = '=';
		}
	}
	/*  Move the logs  */
	while(flag==0){
		pthread_mutex_lock(&mutex);
		for(int i=1;i<ROW;i++){
			if(i%2==1){ //for odd rows, move left
				for(int j=0;j<15;j++){
					map[i][(log_start[i-1]-1+j+COLUMN-1)%(COLUMN-1)] = '=';
				}
				map[i][(log_start[i-1]+14)%(COLUMN-1)] = ' ';
				log_start[i-1] = (log_start[i-1]-1+COLUMN-1)%(COLUMN-1);
			}
			else{// for even rows, move right
				for(int j=0;j<15;j++){
					map[i][(log_start[i-1]+j)%(COLUMN-1)] = '=';
				}
				map[i][(log_start[i-1])%(COLUMN-1)] = ' ';
				log_start[i-1] = (log_start[i-1]+1)%(COLUMN-1);
			}
		}
		if(frog.x!=0&&frog.x!=ROW){
			if(map[frog.x][frog.y]==' '){
				flag=-1;
				printf("\033[H\033[2J");
				printf("You lose the game!!\n");
				break;
			}
			if(frog.x%2==1){
				map[frog.x][frog.y]='0';
				frog.y--;
			}
			else{
				map[frog.x][frog.y]='0';
				frog.y++;
			}
			if(frog.y<-1||frog.y>COLUMN-1){
				flag=-1;
				printf("\033[H\033[2J");
				printf("You lose the game!!\n");
				break;
			}
		}
		printf("\033[H\033[2J");
		for(int i = 0; i <= ROW; ++i)	
			puts( map[i] );
		pthread_mutex_unlock(&mutex);
		usleep(usleep_time);
	}
	pthread_exit(NULL);
}

void *frog_move(void *t){
	while(flag==0){
		if(kbhit()){
			char dir=getchar();
			if(dir=='w'||dir=='W'){
				frog.x--;
				if(frog.x!=ROW){
					map[ROW][frog.y] = '|';
				}
				if(frog.x==0){
					flag=1;
					printf("\033[H\033[2J");
					printf("You win the game!!\n");
					break;
				}
			}
			else if(dir=='a'||dir=='A'){
				frog.y--;
				if(frog.x==ROW){
					map[ROW][frog.y+1] = '|';
					map[ROW][frog.y] = '0';
				}
				if(frog.y==0){
					flag=-1;
					printf("\033[H\033[2J");
					printf("You lose the game!!\n");
					break;
				}
			}
			else if(dir=='d'||dir=='D'){
				frog.y++;
				if(frog.x==ROW){
					map[ROW][frog.y-1] = '|';
					map[ROW][frog.y] = '0';
				}
				if(frog.y==COLUMN-1){
					flag=-1;
					printf("\033[H\033[2J");
					printf("You lose the game!!\n");
					break;
				}
			}
			else if(dir=='s'||dir=='S'){
				frog.x++;
				if(frog.x==ROW){
					map[ROW][frog.y] = '0';
				}
			}
			else if(dir=='q'||dir=='Q'){
				flag=2;
				printf("\033[H\033[2J");
				printf("You exit the game.\n");
				break;
			}
		}
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	pthread_t log_thread, frog_thread;
	pthread_mutex_init(&mutex, NULL);
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */
	int log_thread_id = pthread_create(&log_thread, NULL, logs_move, NULL);
	int frog_thread_id = pthread_create(&frog_thread, NULL, frog_move, NULL);
	pthread_join(log_thread,NULL);
	pthread_join(frog_thread,NULL);
	/*  Display the output for user: win, lose or quit.  */
	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
	return 0;

}
