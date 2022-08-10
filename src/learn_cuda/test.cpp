#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{

  char* output=(char*)malloc(1024*1024*128);
  sprintf(output, "%s", "same");
  sprintf(output+4, "%f", 12.239f);
  printf("%s\n", output);

int b[2] = {3, 5};
int a[3] = {1,10,100};
int* p = a;
int* q = b;
*q++ = *p++;

// output 10, 5
std::cout<<*p<<" "<<*q << std::endl;

int t = 10 << 3;
int tm = (unsigned int)10 & 0x7;
std::cout<< t << " " << tm << std::endl;
return 0;
}