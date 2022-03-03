
set -xe
DIR=$1
FILE=$2
clang -S -emit-llvm $DIR/$FILE.c -Xclang -disable-O0-optnone -o $DIR/$FILE.ll
opt -S -polly-canonicalize $DIR/$FILE.ll -o $DIR/$FILE.preopt.ll
opt -basicaa -polly-ast -analyze $DIR/$FILE.preopt.ll -polly-process-unprofitable -polly-use-llvm-names > $DIR/$FILE.ast
opt -polly-use-llvm-names -basicaa -polly-scops -analyze $DIR/$FILE.preopt.ll -polly-process-unprofitable > $DIR/$FILE.polly_scop
opt -basicaa -polly-use-llvm-names -polly-dependences -analyze $DIR/$FILE.preopt.ll -polly-process-unprofitable > $DIR/$FILE.polly_dep
opt -basicaa -polly-use-llvm-names -polly-export-jscop $DIR/$FILE.preopt.ll -polly-process-unprofitable
opt -basicaa -polly-use-llvm-names $DIR/$FILE.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged -polly-ast -analyze -polly-process-unprofitable

