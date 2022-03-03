; ModuleID = 'lstm_fusion_interchange.c'
source_filename = "lstm_fusion_interchange.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@num_hidden = dso_local constant i32 256, align 4
@num_gate = dso_local constant i32 4, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @lstm_cell_fusion_interchange_fusion(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
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
  store float* %0, float** %7, align 8
  store float* %1, float** %8, align 8
  store float* %2, float** %9, align 8
  store [256 x [256 x float]]* %3, [256 x [256 x float]]** %10, align 8
  store [256 x [256 x float]]* %4, [256 x [256 x float]]** %11, align 8
  store float* %5, float** %12, align 8
  store i32 0, i32* %16, align 4
  br label %19

19:                                               ; preds = %171, %6
  %20 = load i32, i32* %16, align 4
  %21 = icmp slt i32 %20, 256
  br i1 %21, label %22, label %174

22:                                               ; preds = %19
  store i32 0, i32* %17, align 4
  br label %23

23:                                               ; preds = %124, %22
  %24 = load i32, i32* %17, align 4
  %25 = icmp slt i32 %24, 4
  br i1 %25, label %26, label %127

26:                                               ; preds = %23
  %27 = load i32, i32* %17, align 4
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %28
  %30 = load i32, i32* %16, align 4
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds [256 x float], [256 x float]* %29, i64 0, i64 %31
  store float 0.000000e+00, float* %32, align 4
  %33 = load i32, i32* %17, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %34
  %36 = load i32, i32* %16, align 4
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [256 x float], [256 x float]* %35, i64 0, i64 %37
  store float 0.000000e+00, float* %38, align 4
  store i32 0, i32* %18, align 4
  br label %39

39:                                               ; preds = %93, %26
  %40 = load i32, i32* %18, align 4
  %41 = icmp slt i32 %40, 256
  br i1 %41, label %42, label %96

42:                                               ; preds = %39
  %43 = load float*, float** %7, align 8
  %44 = load i32, i32* %18, align 4
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds float, float* %43, i64 %45
  %47 = load float, float* %46, align 4
  %48 = load [256 x [256 x float]]*, [256 x [256 x float]]** %10, align 8
  %49 = load i32, i32* %17, align 4
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %48, i64 %50
  %52 = load i32, i32* %16, align 4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %51, i64 0, i64 %53
  %55 = load i32, i32* %18, align 4
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds [256 x float], [256 x float]* %54, i64 0, i64 %56
  %58 = load float, float* %57, align 4
  %59 = fmul float %47, %58
  %60 = load i32, i32* %17, align 4
  %61 = sext i32 %60 to i64
  %62 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %61
  %63 = load i32, i32* %16, align 4
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds [256 x float], [256 x float]* %62, i64 0, i64 %64
  %66 = load float, float* %65, align 4
  %67 = fadd float %66, %59
  store float %67, float* %65, align 4
  %68 = load float*, float** %9, align 8
  %69 = load i32, i32* %18, align 4
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds float, float* %68, i64 %70
  %72 = load float, float* %71, align 4
  %73 = load [256 x [256 x float]]*, [256 x [256 x float]]** %10, align 8
  %74 = load i32, i32* %17, align 4
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %73, i64 %75
  %77 = load i32, i32* %16, align 4
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %76, i64 0, i64 %78
  %80 = load i32, i32* %18, align 4
  %81 = sext i32 %80 to i64
  %82 = getelementptr inbounds [256 x float], [256 x float]* %79, i64 0, i64 %81
  %83 = load float, float* %82, align 4
  %84 = fmul float %72, %83
  %85 = load i32, i32* %17, align 4
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %86
  %88 = load i32, i32* %16, align 4
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds [256 x float], [256 x float]* %87, i64 0, i64 %89
  %91 = load float, float* %90, align 4
  %92 = fadd float %91, %84
  store float %92, float* %90, align 4
  br label %93

93:                                               ; preds = %42
  %94 = load i32, i32* %18, align 4
  %95 = add nsw i32 %94, 1
  store i32 %95, i32* %18, align 4
  br label %39

96:                                               ; preds = %39
  %97 = load i32, i32* %17, align 4
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %98
  %100 = load i32, i32* %16, align 4
  %101 = sext i32 %100 to i64
  %102 = getelementptr inbounds [256 x float], [256 x float]* %99, i64 0, i64 %101
  %103 = load float, float* %102, align 4
  %104 = load i32, i32* %17, align 4
  %105 = sext i32 %104 to i64
  %106 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %105
  %107 = load i32, i32* %16, align 4
  %108 = sext i32 %107 to i64
  %109 = getelementptr inbounds [256 x float], [256 x float]* %106, i64 0, i64 %108
  %110 = load float, float* %109, align 4
  %111 = fadd float %103, %110
  %112 = load float*, float** %12, align 8
  %113 = load i32, i32* %16, align 4
  %114 = sext i32 %113 to i64
  %115 = getelementptr inbounds float, float* %112, i64 %114
  %116 = load float, float* %115, align 4
  %117 = fadd float %111, %116
  %118 = load i32, i32* %17, align 4
  %119 = sext i32 %118 to i64
  %120 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 %119
  %121 = load i32, i32* %16, align 4
  %122 = sext i32 %121 to i64
  %123 = getelementptr inbounds [256 x float], [256 x float]* %120, i64 0, i64 %122
  store float %117, float* %123, align 4
  br label %124

124:                                              ; preds = %96
  %125 = load i32, i32* %17, align 4
  %126 = add nsw i32 %125, 1
  store i32 %126, i32* %17, align 4
  br label %23

127:                                              ; preds = %23
  %128 = load float*, float** %8, align 8
  %129 = load i32, i32* %16, align 4
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %128, i64 %130
  %132 = load float, float* %131, align 4
  %133 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 2
  %134 = load i32, i32* %16, align 4
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds [256 x float], [256 x float]* %133, i64 0, i64 %135
  %137 = load float, float* %136, align 4
  %138 = fadd float %137, 1.000000e+00
  %139 = fmul float %132, %138
  %140 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 0
  %141 = load i32, i32* %16, align 4
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds [256 x float], [256 x float]* %140, i64 0, i64 %142
  %144 = load float, float* %143, align 4
  %145 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 1
  %146 = load i32, i32* %16, align 4
  %147 = sext i32 %146 to i64
  %148 = getelementptr inbounds [256 x float], [256 x float]* %145, i64 0, i64 %147
  %149 = load float, float* %148, align 4
  %150 = fmul float %144, %149
  %151 = fadd float %139, %150
  %152 = load float*, float** %8, align 8
  %153 = load i32, i32* %16, align 4
  %154 = sext i32 %153 to i64
  %155 = getelementptr inbounds float, float* %152, i64 %154
  store float %151, float* %155, align 4
  %156 = load float*, float** %8, align 8
  %157 = load i32, i32* %16, align 4
  %158 = sext i32 %157 to i64
  %159 = getelementptr inbounds float, float* %156, i64 %158
  %160 = load float, float* %159, align 4
  %161 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 3
  %162 = load i32, i32* %16, align 4
  %163 = sext i32 %162 to i64
  %164 = getelementptr inbounds [256 x float], [256 x float]* %161, i64 0, i64 %163
  %165 = load float, float* %164, align 4
  %166 = fmul float %160, %165
  %167 = load float*, float** %9, align 8
  %168 = load i32, i32* %16, align 4
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds float, float* %167, i64 %169
  store float %166, float* %170, align 4
  br label %171

171:                                              ; preds = %127
  %172 = load i32, i32* %16, align 4
  %173 = add nsw i32 %172, 1
  store i32 %173, i32* %16, align 4
  br label %19

174:                                              ; preds = %19
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
