#include <iostream>
#include <string>
#include <sstream>

// template<typename T>
// class Person{
//   public:
//     int age;
//     T id;
//     int id;
//     Person(){
//       this->age = 10;
//     }
//     Person(int a){
//       this->age=a;
//     }
//     Person(int a, T i){
//       this->age=a;
//       this->id=i;
//     }
//     virtual int inc_age(int inc){
//       return this->age + inc;
//     }
// };


// template<typename T>
// class Student: public Person{
//   public:
  
//   std::string school_name;
//   Student(std::string s){
//     this->school_name = s;
//   }

//   int inc_student_age(int inc){
//     return Person::inc_age(inc);
//   }

//   int inc_age(std::string s){
//     int inc; 
//     std::istringstream ( s ) >> inc;
//     return Person::inc_age(inc);
//   }
// };


class Person{
  public:
    int age;
    int id;
    Person(){
      this->age = 10;
    }
    Person(int a){
      this->age=a;
    }
    Person(int a, int i){
      this->age=a;
      this->id=i;
    }
    virtual int inc_age(int inc){
      return this->age + inc;
    }
};


class Student: public Person{
  public:
  
  std::string school_name;
  Student(std::string s){
    this->school_name = s;
  }

  int inc_student_age(int inc){
    return Person::inc_age(inc);
  }
  // using Person::inc_age;
  int inc_age(std::string s){
    int inc; 
    std::istringstream ( s ) >> inc;
    // printf("");
    return Person::inc_age(inc);
  }
};


int main(){
  std::string s("123");
  // Student<int> stu(s);
  Student stu(s);
  printf("%d\n", stu.inc_student_age(3));
  printf("%d\n", stu.inc_age(s));
  // printf("%d\n", stu.inc_age(5));// 
  printf("%d\n", ((Person*)(&stu))->inc_age((int)5));
  return 0;
}