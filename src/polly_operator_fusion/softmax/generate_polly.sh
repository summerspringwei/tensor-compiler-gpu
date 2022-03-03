
set -xe

FILE=$1
clang -S -emit-llvm $FILE.c -Xclang -disable-O0-optnone -o $FILE.ll
opt -S -polly-canonicalize $FILE.ll -o $FILE.preopt.ll
opt -basicaa -polly-ast -analyze $FILE.preopt.ll -polly-process-unprofitable -polly-use-llvm-names > $FILE.ast
opt -polly-use-llvm-names -basicaa -polly-scops -analyze $FILE.preopt.ll -polly-process-unprofitable > $FILE.polly_scop
opt -basicaa -polly-use-llvm-names -polly-dependences -analyze $FILE.preopt.ll -polly-process-unprofitable > $FILE.polly_dep
opt -basicaa -polly-use-llvm-names -polly-export-jscop $FILE.preopt.ll -polly-process-unprofitable
opt -basicaa -polly-use-llvm-names $FILE.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged -polly-ast -analyze -polly-process-unprofitable

