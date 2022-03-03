// 0. What Relu is? Why activation function needs to be non-linear?

// 1. How would you implement a fully-connected layer (just give a sketch of your approach)?

// 2. Mat Mal: A[n,k]*B[k,m]=C[n,m]. Flops? Bytes transfered? Is it mem or compute bound?
// Flops: n*k*m*2 ;Bytes: (n*k+k*m+n*m)*sizeof(Type)+
// Mat mul is mostly memory-bound


// Compiled with: g++ -Wall -std=c++14 -pthread


// 3. Implement a fast memory allocator.


#include <iostream>

using namespace std;
/*
struct mem_buk{
 
    size_t start_address;
    size_t size;
    bool used;
}

const bulk_size = {1,2,4,8};
std::unordered_map<size_t, std::vector<mem_bulk>> global_pool;

void* my_allocator(size_t num_of_bytes){
        size_t aligned_size = round_up(num_of_byte);
        auto& bulks = global_pool[aligned_size];
    mem_buk m;
    
    bool find = false;
    for(auto b: bulks){
        if(!b.used){
            find = true;
            b.used = true;
            return b.start_address;
        }
    }
    if(!find){
        size_t bigger_size = round_up(aligned_size);
        auto biger_buls = global_pool[bigger_size];
        for(auto b: biger_buls){
        if(!b.used){
            find = true;
            b.used = true;
            // split the biger one to many 
            
            return b.start_address;
        }
    }
    }
    else{
     return nullptr;   
    }
}

void my_free(void* ptr)
*/

class A {
    public: 
    A() : a(5), b(1), c(26.0) {}
    
    public:
    int a;
    short b;
    double c;
};
    
int main(){
    cout << "Hello, World!" << endl;
    //A* a = new A();
    
    void* buff = malloc(sizeof(A)); //< will not compile
    A* a = new(buff) A();
    cout << a->c; //< What does it prints?
    
    
    
    return 0;
}

