#include <iostream>

int main(int argc, char** argv)
{
int b[2] = {3, 5};
int a[3] = {1,10,100};
int* p = a;
int* q = b;
*q++ = *p++;

std::cout<<*p<<" "<<*q;
return 0;
}