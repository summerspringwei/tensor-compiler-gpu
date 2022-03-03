; ModuleID = 'maxpool/maxpool.c'
source_filename = "maxpool/maxpool.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@n = dso_local constant i32 1, align 4
@c = dso_local constant i32 16, align 4
@h = dso_local constant i32 32, align 4
@w = dso_local constant i32 32, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @maxpool([16 x [32 x [32 x float]]]* %0, [16 x [32 x [32 x float]]]* %1) #0 {
  %3 = alloca [16 x [32 x [32 x float]]]*, align 8
  %4 = alloca [16 x [32 x [32 x float]]]*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca float, align 4
  store [16 x [32 x [32 x float]]]* %0, [16 x [32 x [32 x float]]]** %3, align 8
  store [16 x [32 x [32 x float]]]* %1, [16 x [32 x [32 x float]]]** %4, align 8
  store i32 0, i32* %5, align 4
  br label %14

14:                                               ; preds = %107, %2
  %15 = load i32, i32* %5, align 4
  %16 = icmp slt i32 %15, 1
  br i1 %16, label %17, label %110

17:                                               ; preds = %14
  store i32 0, i32* %6, align 4
  br label %18

18:                                               ; preds = %103, %17
  %19 = load i32, i32* %6, align 4
  %20 = icmp slt i32 %19, 16
  br i1 %20, label %21, label %106

21:                                               ; preds = %18
  store i32 0, i32* %7, align 4
  br label %22

22:                                               ; preds = %99, %21
  %23 = load i32, i32* %7, align 4
  %24 = icmp slt i32 %23, 32
  br i1 %24, label %25, label %102

25:                                               ; preds = %22
  store i32 0, i32* %8, align 4
  br label %26

26:                                               ; preds = %95, %25
  %27 = load i32, i32* %8, align 4
  %28 = icmp slt i32 %27, 32
  br i1 %28, label %29, label %98

29:                                               ; preds = %26
  store i32 0, i32* %9, align 4
  br label %30

30:                                               ; preds = %91, %29
  %31 = load i32, i32* %9, align 4
  %32 = icmp slt i32 %31, 3
  br i1 %32, label %33, label %94

33:                                               ; preds = %30
  store i32 0, i32* %10, align 4
  br label %34

34:                                               ; preds = %87, %33
  %35 = load i32, i32* %10, align 4
  %36 = icmp slt i32 %35, 3
  br i1 %36, label %37, label %90

37:                                               ; preds = %34
  %38 = load i32, i32* %7, align 4
  %39 = sub nsw i32 %38, 1
  %40 = load i32, i32* %9, align 4
  %41 = add nsw i32 %39, %40
  store i32 %41, i32* %11, align 4
  %42 = load i32, i32* %8, align 4
  %43 = sub nsw i32 %42, 1
  %44 = load i32, i32* %10, align 4
  %45 = add nsw i32 %43, %44
  store i32 %45, i32* %12, align 4
  store float -1.000000e+04, float* %13, align 4
  %46 = load i32, i32* %11, align 4
  %47 = icmp sge i32 %46, 0
  br i1 %47, label %48, label %72

48:                                               ; preds = %37
  %49 = load i32, i32* %11, align 4
  %50 = icmp slt i32 %49, 32
  br i1 %50, label %51, label %72

51:                                               ; preds = %48
  %52 = load i32, i32* %12, align 4
  %53 = icmp sge i32 %52, 0
  br i1 %53, label %54, label %72

54:                                               ; preds = %51
  %55 = load i32, i32* %12, align 4
  %56 = icmp slt i32 %55, 32
  br i1 %56, label %57, label %72

57:                                               ; preds = %54
  %58 = load [16 x [32 x [32 x float]]]*, [16 x [32 x [32 x float]]]** %3, align 8
  %59 = load i32, i32* %5, align 4
  %60 = sext i32 %59 to i64
  %61 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %58, i64 %60
  %62 = load i32, i32* %6, align 4
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %61, i64 0, i64 %63
  %65 = load i32, i32* %7, align 4
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* %64, i64 0, i64 %66
  %68 = load i32, i32* %8, align 4
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds [32 x float], [32 x float]* %67, i64 0, i64 %69
  %71 = load float, float* %70, align 4
  store float %71, float* %13, align 4
  br label %72

72:                                               ; preds = %57, %54, %51, %48, %37
  %73 = load float, float* %13, align 4
  %74 = load [16 x [32 x [32 x float]]]*, [16 x [32 x [32 x float]]]** %4, align 8
  %75 = load i32, i32* %5, align 4
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %74, i64 %76
  %78 = load i32, i32* %6, align 4
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %77, i64 0, i64 %79
  %81 = load i32, i32* %7, align 4
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* %80, i64 0, i64 %82
  %84 = load i32, i32* %8, align 4
  %85 = sext i32 %84 to i64
  %86 = getelementptr inbounds [32 x float], [32 x float]* %83, i64 0, i64 %85
  store float %73, float* %86, align 4
  br label %87

87:                                               ; preds = %72
  %88 = load i32, i32* %10, align 4
  %89 = add nsw i32 %88, 1
  store i32 %89, i32* %10, align 4
  br label %34

90:                                               ; preds = %34
  br label %91

91:                                               ; preds = %90
  %92 = load i32, i32* %9, align 4
  %93 = add nsw i32 %92, 1
  store i32 %93, i32* %9, align 4
  br label %30

94:                                               ; preds = %30
  br label %95

95:                                               ; preds = %94
  %96 = load i32, i32* %8, align 4
  %97 = add nsw i32 %96, 1
  store i32 %97, i32* %8, align 4
  br label %26

98:                                               ; preds = %26
  br label %99

99:                                               ; preds = %98
  %100 = load i32, i32* %7, align 4
  %101 = add nsw i32 %100, 1
  store i32 %101, i32* %7, align 4
  br label %22

102:                                              ; preds = %22
  br label %103

103:                                              ; preds = %102
  %104 = load i32, i32* %6, align 4
  %105 = add nsw i32 %104, 1
  store i32 %105, i32* %6, align 4
  br label %18

106:                                              ; preds = %18
  br label %107

107:                                              ; preds = %106
  %108 = load i32, i32* %5, align 4
  %109 = add nsw i32 %108, 1
  store i32 %109, i32* %5, align 4
  br label %14

110:                                              ; preds = %14
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
