
#include <stdio.h>
#include "learn_namespace.h"
#include "learn_namespace.h"
#include "learn_namespace.h"
using namespace souffle;

int add(int a, int b) {
    using namespace souffle::space1;
    const int kOff = A + B;
    return a + b + kOff + D;
}

int sum(int a, int b) {
    using namespace souffle::space2;
    const int kOff = A + B;
    return a + b + kOff + D;
}


int main(int argc, char* argv[]) {
    printf("add: %d\n", add(1, 2));
    sum(1, 2);
    printf("sum: %d\n", sum(1, 2));
    return 0;
}