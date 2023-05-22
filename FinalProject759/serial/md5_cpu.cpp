#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sstream>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define CONST_WORD_LIMIT 10
#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET))

#include "md5.h"

/* Global variables */
char g_cracked[CONST_WORD_LIMIT];

void next(int pass, int len, char conv_pass[]){
	int i = len - 1;
	int j;
	int point;
	for (j =0 ; j < len ; j++){
		conv_pass[j] = 'a';
	}
	conv_pass[len] = '\0';
	while(1){
		if (pass >0){
			point = pass % CONST_CHARSET_LENGTH;
			conv_pass[i] = CONST_CHARSET[point];
			i--;
			pass = pass / CONST_CHARSET_LENGTH;
		}else{
			break;
		}
	}
}

void decrypt_pass(int begin, int end, int len, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04, bool &found){
	char pas[len];
	while(1){
		next(begin, len, pas);
    uint32_t resHash01, resHash02, resHash03, resHash04;
    getMd5Hash((unsigned char *)pas, len, &resHash01, &resHash02, &resHash03, &resHash04);
    if (resHash01 == hash01 && resHash02 == hash02 && resHash03 == hash03 && resHash04 == hash04)
    {
      memcpy(g_cracked, pas, len);
      found = true;
      break;
		}else if(begin == end){
			break;
		}else{
			begin ++;
		}
	}
	return;
}

int main(int argc, char *argv[])
{
  /* Check arguments */
  if (argc != 3 || strlen(argv[1]) != 32)
  {
    std::cout << argv[0] << " <md5_hash> <pass_len>" << std::endl;
    return -1;
  }

  const unsigned long PASS_LEN = atoi(argv[2]);

  uint32_t md5Hash[4];

  /* Parse hash */
  for (uint8_t i = 0; i < 4; i++)
  {
    char tmp[16];

    strncpy(tmp, argv[1] + i * 8, 8);
    sscanf(tmp, "%x", &md5Hash[i]);
    md5Hash[i] = (md5Hash[i] & 0xFF000000) >> 24 | (md5Hash[i] & 0x00FF0000) >> 8 | (md5Hash[i] & 0x0000FF00) << 8 | (md5Hash[i] & 0x000000FF) << 24;
  }

  int total_password = pow(CONST_CHARSET_LENGTH, PASS_LEN);
  bool found = false;
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_msec;
  start = high_resolution_clock::now();
  decrypt_pass(0, total_password-1, PASS_LEN, md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3], found);
  end = high_resolution_clock::now();
  duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  std::cout << duration_msec.count() << std::endl;
  if (found) {
      std::cout << "cracked: " << g_cracked << std::endl;
  } else {
    std::cout << "not found" << std::endl;
  }

  return 0;
}