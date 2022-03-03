; ModuleID = 'stencil/stencil2d.c'
source_filename = "stencil/stencil2d.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@L = dso_local constant i32 10, align 4
@M = dso_local constant i32 10, align 4
@N = dso_local constant i32 10, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @stencil2d([10 x float]* %0) #0 {
  %2 = alloca [10 x float]*, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store [10 x float]* %0, [10 x float]** %2, align 8
  store i32 0, i32* %3, align 4
  br label %6

6:                                                ; preds = %75, %1
  %7 = load i32, i32* %3, align 4
  %8 = icmp slt i32 %7, 10
  br i1 %8, label %9, label %78

9:                                                ; preds = %6
  store i32 1, i32* %4, align 4
  br label %10

10:                                               ; preds = %71, %9
  %11 = load i32, i32* %4, align 4
  %12 = icmp slt i32 %11, 9
  br i1 %12, label %13, label %74

13:                                               ; preds = %10
  store i32 1, i32* %5, align 4
  br label %14

14:                                               ; preds = %67, %13
  %15 = load i32, i32* %5, align 4
  %16 = icmp slt i32 %15, 9
  br i1 %16, label %17, label %70

17:                                               ; preds = %14
  %18 = load [10 x float]*, [10 x float]** %2, align 8
  %19 = load i32, i32* %4, align 4
  %20 = add nsw i32 %19, 1
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [10 x float], [10 x float]* %18, i64 %21
  %23 = load i32, i32* %5, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [10 x float], [10 x float]* %22, i64 0, i64 %24
  %26 = load float, float* %25, align 4
  %27 = load [10 x float]*, [10 x float]** %2, align 8
  %28 = load i32, i32* %4, align 4
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds [10 x float], [10 x float]* %27, i64 %29
  %31 = load i32, i32* %5, align 4
  %32 = add nsw i32 %31, 1
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds [10 x float], [10 x float]* %30, i64 0, i64 %33
  %35 = load float, float* %34, align 4
  %36 = fadd float %26, %35
  %37 = load [10 x float]*, [10 x float]** %2, align 8
  %38 = load i32, i32* %4, align 4
  %39 = sub nsw i32 %38, 1
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [10 x float], [10 x float]* %37, i64 %40
  %42 = load i32, i32* %5, align 4
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds [10 x float], [10 x float]* %41, i64 0, i64 %43
  %45 = load float, float* %44, align 4
  %46 = fadd float %36, %45
  %47 = load [10 x float]*, [10 x float]** %2, align 8
  %48 = load i32, i32* %4, align 4
  %49 = sext i32 %48 to i64
  %50 = getelementptr inbounds [10 x float], [10 x float]* %47, i64 %49
  %51 = load i32, i32* %5, align 4
  %52 = sub nsw i32 %51, 1
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [10 x float], [10 x float]* %50, i64 0, i64 %53
  %55 = load float, float* %54, align 4
  %56 = fadd float %46, %55
  %57 = fpext float %56 to double
  %58 = fmul double %57, 2.500000e-01
  %59 = fptrunc double %58 to float
  %60 = load [10 x float]*, [10 x float]** %2, align 8
  %61 = load i32, i32* %4, align 4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds [10 x float], [10 x float]* %60, i64 %62
  %64 = load i32, i32* %5, align 4
  %65 = sext i32 %64 to i64
  %66 = getelementptr inbounds [10 x float], [10 x float]* %63, i64 0, i64 %65
  store float %59, float* %66, align 4
  br label %67

67:                                               ; preds = %17
  %68 = load i32, i32* %5, align 4
  %69 = add nsw i32 %68, 1
  store i32 %69, i32* %5, align 4
  br label %14

70:                                               ; preds = %14
  br label %71

71:                                               ; preds = %70
  %72 = load i32, i32* %4, align 4
  %73 = add nsw i32 %72, 1
  store i32 %73, i32* %4, align 4
  br label %10

74:                                               ; preds = %10
  br label %75

75:                                               ; preds = %74
  %76 = load i32, i32* %3, align 4
  %77 = add nsw i32 %76, 1
  store i32 %77, i32* %3, align 4
  br label %6

78:                                               ; preds = %6
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [10 x [10 x float]], align 16
  store i32 0, i32* %1, align 4
  %3 = getelementptr inbounds [10 x [10 x float]], [10 x [10 x float]]* %2, i64 0, i64 0
  call void @stencil2d([10 x float]* %3)
  ret i32 0
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
