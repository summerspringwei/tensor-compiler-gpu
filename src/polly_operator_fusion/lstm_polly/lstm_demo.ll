; ModuleID = 'lstm_demo.cpp'
source_filename = "lstm_demo.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @_Z9lstm_cellPfS_S_PA256_A256_fS2_S_(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  %9 = alloca float*, align 8
  %10 = alloca [256 x [256 x float]]*, align 8
  %11 = alloca [256 x [256 x float]]*, align 8
  %12 = alloca float*, align 8
  %13 = alloca [4 x [256 x float]], align 16
  %14 = alloca [4 x [256 x float]], align 16
  %15 = alloca [4 x [256 x float]], align 16
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  store float* %0, float** %7, align 8
  store float* %1, float** %8, align 8
  store float* %2, float** %9, align 8
  store [256 x [256 x float]]* %3, [256 x [256 x float]]** %10, align 8
  store [256 x [256 x float]]* %4, [256 x [256 x float]]** %11, align 8
  store float* %5, float** %12, align 8
  store i32 0, i32* %16, align 4
  br label %20

20:                                               ; preds = %129, %6
  %21 = load i32, i32* %16, align 4
  %22 = icmp slt i32 %21, 4
  br i1 %22, label %23, label %132

23:                                               ; preds = %20
  store i32 0, i32* %17, align 4
  br label %24

24:                                               ; preds = %125, %23
  %25 = load i32, i32* %17, align 4
  %26 = icmp slt i32 %25, 256
  br i1 %26, label %27, label %128

27:                                               ; preds = %24
  %28 = load i32, i32* %16, align 4
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %29
  %31 = load i32, i32* %17, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [256 x float], [256 x float]* %30, i64 0, i64 %32
  store float 0.000000e+00, float* %33, align 4
  %34 = load i32, i32* %16, align 4
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %35
  %37 = load i32, i32* %17, align 4
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds [256 x float], [256 x float]* %36, i64 0, i64 %38
  store float 0.000000e+00, float* %39, align 4
  store i32 0, i32* %18, align 4
  br label %40

40:                                               ; preds = %94, %27
  %41 = load i32, i32* %18, align 4
  %42 = icmp slt i32 %41, 256
  br i1 %42, label %43, label %97

43:                                               ; preds = %40
  %44 = load float*, float** %7, align 8
  %45 = load i32, i32* %18, align 4
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds float, float* %44, i64 %46
  %48 = load float, float* %47, align 4
  %49 = load [256 x [256 x float]]*, [256 x [256 x float]]** %10, align 8
  %50 = load i32, i32* %16, align 4
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %49, i64 %51
  %53 = load i32, i32* %17, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %52, i64 0, i64 %54
  %56 = load i32, i32* %18, align 4
  %57 = sext i32 %56 to i64
  %58 = getelementptr inbounds [256 x float], [256 x float]* %55, i64 0, i64 %57
  %59 = load float, float* %58, align 4
  %60 = fmul float %48, %59
  %61 = load i32, i32* %16, align 4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %62
  %64 = load i32, i32* %17, align 4
  %65 = sext i32 %64 to i64
  %66 = getelementptr inbounds [256 x float], [256 x float]* %63, i64 0, i64 %65
  %67 = load float, float* %66, align 4
  %68 = fadd float %67, %60
  store float %68, float* %66, align 4
  %69 = load float*, float** %7, align 8
  %70 = load i32, i32* %18, align 4
  %71 = sext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %69, i64 %71
  %73 = load float, float* %72, align 4
  %74 = load [256 x [256 x float]]*, [256 x [256 x float]]** %10, align 8
  %75 = load i32, i32* %16, align 4
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %74, i64 %76
  %78 = load i32, i32* %17, align 4
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %77, i64 0, i64 %79
  %81 = load i32, i32* %18, align 4
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [256 x float], [256 x float]* %80, i64 0, i64 %82
  %84 = load float, float* %83, align 4
  %85 = fmul float %73, %84
  %86 = load i32, i32* %16, align 4
  %87 = sext i32 %86 to i64
  %88 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %87
  %89 = load i32, i32* %17, align 4
  %90 = sext i32 %89 to i64
  %91 = getelementptr inbounds [256 x float], [256 x float]* %88, i64 0, i64 %90
  %92 = load float, float* %91, align 4
  %93 = fadd float %92, %85
  store float %93, float* %91, align 4
  br label %94

94:                                               ; preds = %43
  %95 = load i32, i32* %18, align 4
  %96 = add nsw i32 %95, 1
  store i32 %96, i32* %18, align 4
  br label %40

97:                                               ; preds = %40
  %98 = load i32, i32* %16, align 4
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %99
  %101 = load i32, i32* %17, align 4
  %102 = sext i32 %101 to i64
  %103 = getelementptr inbounds [256 x float], [256 x float]* %100, i64 0, i64 %102
  %104 = load float, float* %103, align 4
  %105 = load i32, i32* %16, align 4
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %106
  %108 = load i32, i32* %17, align 4
  %109 = sext i32 %108 to i64
  %110 = getelementptr inbounds [256 x float], [256 x float]* %107, i64 0, i64 %109
  %111 = load float, float* %110, align 4
  %112 = fadd float %104, %111
  %113 = load float*, float** %12, align 8
  %114 = load i32, i32* %17, align 4
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds float, float* %113, i64 %115
  %117 = load float, float* %116, align 4
  %118 = fadd float %112, %117
  %119 = load i32, i32* %16, align 4
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 %120
  %122 = load i32, i32* %17, align 4
  %123 = sext i32 %122 to i64
  %124 = getelementptr inbounds [256 x float], [256 x float]* %121, i64 0, i64 %123
  store float %118, float* %124, align 4
  br label %125

125:                                              ; preds = %97
  %126 = load i32, i32* %17, align 4
  %127 = add nsw i32 %126, 1
  store i32 %127, i32* %17, align 4
  br label %24

128:                                              ; preds = %24
  br label %129

129:                                              ; preds = %128
  %130 = load i32, i32* %16, align 4
  %131 = add nsw i32 %130, 1
  store i32 %131, i32* %16, align 4
  br label %20

132:                                              ; preds = %20
  store i32 0, i32* %19, align 4
  br label %133

133:                                              ; preds = %180, %132
  %134 = load i32, i32* %19, align 4
  %135 = icmp slt i32 %134, 256
  br i1 %135, label %136, label %183

136:                                              ; preds = %133
  %137 = load float*, float** %8, align 8
  %138 = load i32, i32* %19, align 4
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds float, float* %137, i64 %139
  %141 = load float, float* %140, align 4
  %142 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 2
  %143 = load i32, i32* %19, align 4
  %144 = sext i32 %143 to i64
  %145 = getelementptr inbounds [256 x float], [256 x float]* %142, i64 0, i64 %144
  %146 = load float, float* %145, align 4
  %147 = fadd float %146, 1.000000e+00
  %148 = fmul float %141, %147
  %149 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 0
  %150 = load i32, i32* %19, align 4
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds [256 x float], [256 x float]* %149, i64 0, i64 %151
  %153 = load float, float* %152, align 4
  %154 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 1
  %155 = load i32, i32* %19, align 4
  %156 = sext i32 %155 to i64
  %157 = getelementptr inbounds [256 x float], [256 x float]* %154, i64 0, i64 %156
  %158 = load float, float* %157, align 4
  %159 = fmul float %153, %158
  %160 = fadd float %148, %159
  %161 = load float*, float** %8, align 8
  %162 = load i32, i32* %19, align 4
  %163 = sext i32 %162 to i64
  %164 = getelementptr inbounds float, float* %161, i64 %163
  store float %160, float* %164, align 4
  %165 = load float*, float** %8, align 8
  %166 = load i32, i32* %19, align 4
  %167 = sext i32 %166 to i64
  %168 = getelementptr inbounds float, float* %165, i64 %167
  %169 = load float, float* %168, align 4
  %170 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 3
  %171 = load i32, i32* %19, align 4
  %172 = sext i32 %171 to i64
  %173 = getelementptr inbounds [256 x float], [256 x float]* %170, i64 0, i64 %172
  %174 = load float, float* %173, align 4
  %175 = fmul float %169, %174
  %176 = load float*, float** %9, align 8
  %177 = load i32, i32* %19, align 4
  %178 = sext i32 %177 to i64
  %179 = getelementptr inbounds float, float* %176, i64 %178
  store float %175, float* %179, align 4
  br label %180

180:                                              ; preds = %136
  %181 = load i32, i32* %19, align 4
  %182 = add nsw i32 %181, 1
  store i32 %182, i32* %19, align 4
  br label %133

183:                                              ; preds = %133
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
