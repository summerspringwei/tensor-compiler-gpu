// opt --polly-use-llvm-names --basicaa --polly-scops --analyze lstm_demo.preopt.ll --polly-process-unprofitable
// Printing analysis 'Basic Alias Analysis (stateless AA impl)' for function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Pass::print not implemented for pass: 'Basic Alias Analysis (stateless AA impl)'!
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '%32 => %49' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Invalid Scop!
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '%13 => %23' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Invalid Scop!
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '%10 => %31' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Invalid Scop!
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '.preheader4 => .preheader' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Invalid Scop!
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '.preheader4 => %49' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
    Function: _Z9lstm_cellPfS_S_PA256_A256_fS2_S_
    Region: %.preheader4---%49
    Max Loop Depth:  3
    Invariant Accesses: {
    }
    Context:
    {  :  }
    Assumed Context:
    {  :  }
    Invalid Context:
    {  : false }
    Arrays {
        float MemRef0[*][256]; // Element size 4
        float MemRef1[*][256]; // Element size 4
        float MemRef2[*]; // Element size 4
        float MemRef3[*][256][256]; // Element size 4
        float MemRef4[*]; // Element size 4
        float MemRef5[*][256]; // Element size 4
        float MemRef6[*]; // Element size 4
        float MemRef7[*]; // Element size 4
    }
    Arrays (Bounds as pw_affs) {
        float MemRef0[*][ { [] -> [(256)] } ]; // Element size 4
        float MemRef1[*][ { [] -> [(256)] } ]; // Element size 4
        float MemRef2[*]; // Element size 4
        float MemRef3[*][ { [] -> [(256)] } ][ { [] -> [(256)] } ]; // Element size 4
        float MemRef4[*]; // Element size 4
        float MemRef5[*][ { [] -> [(256)] } ]; // Element size 4
        float MemRef6[*]; // Element size 4
        float MemRef7[*]; // Element size 4
    }
    Alias Groups (3):
        [[ <{ MemRef2[(0)] }, { MemRef2[(256)] }> <{ MemRef7[(0)] }, { MemRef7[(256)] }> <{ MemRef6[(0)] }, { MemRef6[(256)] }> ]]
        [[ <{ MemRef4[(0)] }, { MemRef4[(256)] }> <{ MemRef7[(0)] }, { MemRef7[(256)] }> <{ MemRef6[(0)] }, { MemRef6[(256)] }> ]]
        [[ <{ MemRef3[(0), (0), (0)] }, { MemRef3[(3), (255), (256)] }> <{ MemRef7[(0)] }, { MemRef7[(256)] }> <{ MemRef6[(0)] }, { MemRef6[(256)] }> ]]
    Statements {
        Stmt1
            Domain :=
                { Stmt1[i0, i1] : 0 <= i0 <= 3 and 0 <= i1 <= 255 };
            Schedule :=
                { Stmt1[i0, i1] -> [0, i0, i1, 0, 0] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt1[i0, i1] -> MemRef0[i0, i1] };
        Stmt1_b
            Domain :=
                { Stmt1_b[i0, i1] : 0 <= i0 <= 3 and 0 <= i1 <= 255 };
            Schedule :=
                { Stmt1_b[i0, i1] -> [0, i0, i1, 1, 0] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt1_b[i0, i1] -> MemRef1[i0, i1] };
        Stmt2
            Domain :=
                { Stmt2[i0, i1, i2] : 0 <= i0 <= 3 and 0 <= i1 <= 255 and 0 <= i2 <= 255 };
            Schedule :=
                { Stmt2[i0, i1, i2] -> [0, i0, i1, 2, i2] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef2[i2] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef3[i0, i1, i2] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef0[i0, i1] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef0[i0, i1] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef1[i0, i1] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt2[i0, i1, i2] -> MemRef1[i0, i1] };
        Stmt3
            Domain :=
                { Stmt3[i0, i1] : 0 <= i0 <= 3 and 0 <= i1 <= 255 };
            Schedule :=
                { Stmt3[i0, i1] -> [0, i0, i1, 3, 0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt3[i0, i1] -> MemRef0[i0, i1] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt3[i0, i1] -> MemRef1[i0, i1] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt3[i0, i1] -> MemRef4[i1] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt3[i0, i1] -> MemRef5[i0, i1] };
        Stmt6
            Domain :=
                { Stmt6[i0] : 0 <= i0 <= 255 };
            Schedule :=
                { Stmt6[i0] -> [1, i0, 0, 0, 0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef6[i0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef5[2, i0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef5[0, i0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef5[1, i0] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef6[i0] };
            ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef5[3, i0] };
            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
                { Stmt6[i0] -> MemRef7[i0] };
    }
// Printing analysis 'Polly - Create polyhedral description of Scops' for region: '%6 => <Function Return>' in function '_Z9lstm_cellPfS_S_PA256_A256_fS2_S_':
// Invalid Scop!