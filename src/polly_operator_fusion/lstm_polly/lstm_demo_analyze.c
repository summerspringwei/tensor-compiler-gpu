// opt --basicaa --polly-ast --analyze lstm_demo.preopt.ll -polly-process-unprofitable --polly-use-llvm-names
// Printing analysis 'Basic Alias Analysis (stateless AA impl)' for function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Pass::print not implemented for pass: 'Basic Alias Analysis (stateless AA impl)'!
// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '%32 => %49' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '%13 => %23' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '%10 => %31' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '.preheader4 => .preheader' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '.preheader4 => %49' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// :: isl ast :: _Z9lstm_cellPfS_S_PA256_A256_fS2_S_ :: %.preheader4---%49

if (1 && (&MemRef6[256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef6[0]) && (&MemRef2[256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef2[0]) && (&MemRef4[256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef4[0]) && (&MemRef3[3][255][256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef3[0][0][0]) && (&MemRef2[256] <= &MemRef6[0] || &MemRef6[256] <= &MemRef2[0]) && (&MemRef4[256] <= &MemRef6[0] || &MemRef6[256] <= &MemRef4[0]) && (&MemRef3[3][255][256] <= &MemRef6[0] || &MemRef6[256] <= &MemRef3[0][0][0]))

    {
      for (int c0 = 0; c0 <= 3; c0 += 1)
        for (int c1 = 0; c1 <= 255; c1 += 1) {
          Stmt1(c0, c1);
          Stmt1_b(c0, c1);
          for (int c2 = 0; c2 <= 255; c2 += 1)
            Stmt2(c0, c1, c2);
          Stmt3(c0, c1);
        }
      for (int c0 = 0; c0 <= 255; c0 += 1)
        Stmt6(c0);
    }

else
    {  /* original code */ }

// Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '%6 => <Function Return>' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':