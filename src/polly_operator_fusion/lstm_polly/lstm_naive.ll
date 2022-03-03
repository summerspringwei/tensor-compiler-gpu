; ModuleID = 'lstm_naive.c'
source_filename = "lstm_naive.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@num_hidden = dso_local constant i32 256, align 4
@num_gate = dso_local constant i32 4, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @lstm_cell_naive(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
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
  %20 = alloca i32, align 4
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  %23 = alloca i32, align 4
  store float* %0, float** %7, align 8
  store float* %1, float** %8, align 8
  store float* %2, float** %9, align 8
  store [256 x [256 x float]]* %3, [256 x [256 x float]]** %10, align 8
  store [256 x [256 x float]]* %4, [256 x [256 x float]]** %11, align 8
  store float* %5, float** %12, align 8
  store i32 0, i32* %16, align 4
  br label %24

24:                                               ; preds = %157, %6
  %25 = load i32, i32* %16, align 4
  %26 = icmp slt i32 %25, 4
  br i1 %26, label %27, label %160

27:                                               ; preds = %24
  store i32 0, i32* %17, align 4
  br label %28

28:                                               ; preds = %71, %27
  %29 = load i32, i32* %17, align 4
  %30 = icmp slt i32 %29, 256
  br i1 %30, label %31, label %74

31:                                               ; preds = %28
  %32 = load i32, i32* %16, align 4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %33
  %35 = load i32, i32* %17, align 4
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [256 x float], [256 x float]* %34, i64 0, i64 %36
  store float 0.000000e+00, float* %37, align 4
  store i32 0, i32* %18, align 4
  br label %38

38:                                               ; preds = %67, %31
  %39 = load i32, i32* %18, align 4
  %40 = icmp slt i32 %39, 256
  br i1 %40, label %41, label %70

41:                                               ; preds = %38
  %42 = load float*, float** %7, align 8
  %43 = load i32, i32* %18, align 4
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds float, float* %42, i64 %44
  %46 = load float, float* %45, align 4
  %47 = load [256 x [256 x float]]*, [256 x [256 x float]]** %10, align 8
  %48 = load i32, i32* %16, align 4
  %49 = sext i32 %48 to i64
  %50 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %47, i64 %49
  %51 = load i32, i32* %17, align 4
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %50, i64 0, i64 %52
  %54 = load i32, i32* %18, align 4
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds [256 x float], [256 x float]* %53, i64 0, i64 %55
  %57 = load float, float* %56, align 4
  %58 = fmul float %46, %57
  %59 = load i32, i32* %16, align 4
  %60 = sext i32 %59 to i64
  %61 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %60
  %62 = load i32, i32* %17, align 4
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds [256 x float], [256 x float]* %61, i64 0, i64 %63
  %65 = load float, float* %64, align 4
  %66 = fadd float %65, %58
  store float %66, float* %64, align 4
  br label %67

67:                                               ; preds = %41
  %68 = load i32, i32* %18, align 4
  %69 = add nsw i32 %68, 1
  store i32 %69, i32* %18, align 4
  br label %38

70:                                               ; preds = %38
  br label %71

71:                                               ; preds = %70
  %72 = load i32, i32* %17, align 4
  %73 = add nsw i32 %72, 1
  store i32 %73, i32* %17, align 4
  br label %28

74:                                               ; preds = %28
  store i32 0, i32* %19, align 4
  br label %75

75:                                               ; preds = %118, %74
  %76 = load i32, i32* %19, align 4
  %77 = icmp slt i32 %76, 256
  br i1 %77, label %78, label %121

78:                                               ; preds = %75
  %79 = load i32, i32* %16, align 4
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %80
  %82 = load i32, i32* %19, align 4
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds [256 x float], [256 x float]* %81, i64 0, i64 %83
  store float 0.000000e+00, float* %84, align 4
  store i32 0, i32* %20, align 4
  br label %85

85:                                               ; preds = %114, %78
  %86 = load i32, i32* %20, align 4
  %87 = icmp slt i32 %86, 256
  br i1 %87, label %88, label %117

88:                                               ; preds = %85
  %89 = load float*, float** %9, align 8
  %90 = load i32, i32* %20, align 4
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds float, float* %89, i64 %91
  %93 = load float, float* %92, align 4
  %94 = load [256 x [256 x float]]*, [256 x [256 x float]]** %11, align 8
  %95 = load i32, i32* %16, align 4
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %94, i64 %96
  %98 = load i32, i32* %19, align 4
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %97, i64 0, i64 %99
  %101 = load i32, i32* %20, align 4
  %102 = sext i32 %101 to i64
  %103 = getelementptr inbounds [256 x float], [256 x float]* %100, i64 0, i64 %102
  %104 = load float, float* %103, align 4
  %105 = fmul float %93, %104
  %106 = load i32, i32* %16, align 4
  %107 = sext i32 %106 to i64
  %108 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %107
  %109 = load i32, i32* %19, align 4
  %110 = sext i32 %109 to i64
  %111 = getelementptr inbounds [256 x float], [256 x float]* %108, i64 0, i64 %110
  %112 = load float, float* %111, align 4
  %113 = fadd float %112, %105
  store float %113, float* %111, align 4
  br label %114

114:                                              ; preds = %88
  %115 = load i32, i32* %20, align 4
  %116 = add nsw i32 %115, 1
  store i32 %116, i32* %20, align 4
  br label %85

117:                                              ; preds = %85
  br label %118

118:                                              ; preds = %117
  %119 = load i32, i32* %19, align 4
  %120 = add nsw i32 %119, 1
  store i32 %120, i32* %19, align 4
  br label %75

121:                                              ; preds = %75
  store i32 0, i32* %21, align 4
  br label %122

122:                                              ; preds = %153, %121
  %123 = load i32, i32* %21, align 4
  %124 = icmp slt i32 %123, 256
  br i1 %124, label %125, label %156

125:                                              ; preds = %122
  %126 = load i32, i32* %16, align 4
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %13, i64 0, i64 %127
  %129 = load i32, i32* %21, align 4
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds [256 x float], [256 x float]* %128, i64 0, i64 %130
  %132 = load float, float* %131, align 4
  %133 = load i32, i32* %16, align 4
  %134 = sext i32 %133 to i64
  %135 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %14, i64 0, i64 %134
  %136 = load i32, i32* %21, align 4
  %137 = sext i32 %136 to i64
  %138 = getelementptr inbounds [256 x float], [256 x float]* %135, i64 0, i64 %137
  %139 = load float, float* %138, align 4
  %140 = fadd float %132, %139
  %141 = load float*, float** %12, align 8
  %142 = load i32, i32* %21, align 4
  %143 = sext i32 %142 to i64
  %144 = getelementptr inbounds float, float* %141, i64 %143
  %145 = load float, float* %144, align 4
  %146 = fadd float %140, %145
  %147 = load i32, i32* %16, align 4
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 %148
  %150 = load i32, i32* %21, align 4
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds [256 x float], [256 x float]* %149, i64 0, i64 %151
  store float %146, float* %152, align 4
  br label %153

153:                                              ; preds = %125
  %154 = load i32, i32* %21, align 4
  %155 = add nsw i32 %154, 1
  store i32 %155, i32* %21, align 4
  br label %122

156:                                              ; preds = %122
  br label %157

157:                                              ; preds = %156
  %158 = load i32, i32* %16, align 4
  %159 = add nsw i32 %158, 1
  store i32 %159, i32* %16, align 4
  br label %24

160:                                              ; preds = %24
  store i32 0, i32* %22, align 4
  br label %161

161:                                              ; preds = %193, %160
  %162 = load i32, i32* %22, align 4
  %163 = icmp slt i32 %162, 256
  br i1 %163, label %164, label %196

164:                                              ; preds = %161
  %165 = load float*, float** %8, align 8
  %166 = load i32, i32* %22, align 4
  %167 = sext i32 %166 to i64
  %168 = getelementptr inbounds float, float* %165, i64 %167
  %169 = load float, float* %168, align 4
  %170 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 2
  %171 = load i32, i32* %22, align 4
  %172 = sext i32 %171 to i64
  %173 = getelementptr inbounds [256 x float], [256 x float]* %170, i64 0, i64 %172
  %174 = load float, float* %173, align 4
  %175 = fadd float %174, 1.000000e+00
  %176 = fmul float %169, %175
  %177 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 0
  %178 = load i32, i32* %22, align 4
  %179 = sext i32 %178 to i64
  %180 = getelementptr inbounds [256 x float], [256 x float]* %177, i64 0, i64 %179
  %181 = load float, float* %180, align 4
  %182 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 1
  %183 = load i32, i32* %22, align 4
  %184 = sext i32 %183 to i64
  %185 = getelementptr inbounds [256 x float], [256 x float]* %182, i64 0, i64 %184
  %186 = load float, float* %185, align 4
  %187 = fmul float %181, %186
  %188 = fadd float %176, %187
  %189 = load float*, float** %8, align 8
  %190 = load i32, i32* %22, align 4
  %191 = sext i32 %190 to i64
  %192 = getelementptr inbounds float, float* %189, i64 %191
  store float %188, float* %192, align 4
  br label %193

193:                                              ; preds = %164
  %194 = load i32, i32* %22, align 4
  %195 = add nsw i32 %194, 1
  store i32 %195, i32* %22, align 4
  br label %161

196:                                              ; preds = %161
  store i32 0, i32* %23, align 4
  br label %197

197:                                              ; preds = %216, %196
  %198 = load i32, i32* %23, align 4
  %199 = icmp slt i32 %198, 256
  br i1 %199, label %200, label %219

200:                                              ; preds = %197
  %201 = load float*, float** %8, align 8
  %202 = load i32, i32* %23, align 4
  %203 = sext i32 %202 to i64
  %204 = getelementptr inbounds float, float* %201, i64 %203
  %205 = load float, float* %204, align 4
  %206 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %15, i64 0, i64 3
  %207 = load i32, i32* %23, align 4
  %208 = sext i32 %207 to i64
  %209 = getelementptr inbounds [256 x float], [256 x float]* %206, i64 0, i64 %208
  %210 = load float, float* %209, align 4
  %211 = fmul float %205, %210
  %212 = load float*, float** %9, align 8
  %213 = load i32, i32* %23, align 4
  %214 = sext i32 %213 to i64
  %215 = getelementptr inbounds float, float* %212, i64 %214
  store float %211, float* %215, align 4
  br label %216

216:                                              ; preds = %200
  %217 = load i32, i32* %23, align 4
  %218 = add nsw i32 %217, 1
  store i32 %218, i32* %23, align 4
  br label %197

219:                                              ; preds = %197
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
