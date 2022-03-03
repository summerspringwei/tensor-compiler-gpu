; ModuleID = 'lstm_demo.ll'
source_filename = "lstm_demo.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @_Z9lstm_cellPfS_S_PA256_A256_fS2_S_(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
  %7 = alloca [4 x [256 x float]], align 16
  %8 = alloca [4 x [256 x float]], align 16
  %9 = alloca [4 x [256 x float]], align 16
  br label %.split

.split:                                           ; preds = %6
  br label %.preheader4

.preheader4:                                      ; preds = %.split, %31
  %indvars.iv15 = phi i64 [ 0, %.split ], [ %indvars.iv.next16, %31 ]
  br label %10

.preheader:                                       ; preds = %31
  br label %32

10:                                               ; preds = %.preheader4, %23
  %indvars.iv12 = phi i64 [ 0, %.preheader4 ], [ %indvars.iv.next13, %23 ]
  %11 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %7, i64 0, i64 %indvars.iv15, i64 %indvars.iv12
  store float 0.000000e+00, float* %11, align 4
  %12 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %8, i64 0, i64 %indvars.iv15, i64 %indvars.iv12
  store float 0.000000e+00, float* %12, align 4
  br label %13

13:                                               ; preds = %10, %13
  %indvars.iv9 = phi i64 [ 0, %10 ], [ %indvars.iv.next10, %13 ]
  %14 = getelementptr inbounds float, float* %0, i64 %indvars.iv9
  %15 = load float, float* %14, align 4
  %16 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %3, i64 %indvars.iv15, i64 %indvars.iv12, i64 %indvars.iv9
  %17 = load float, float* %16, align 4
  %18 = fmul float %15, %17
  %19 = load float, float* %11, align 4
  %20 = fadd float %19, %18
  store float %20, float* %11, align 4
  %21 = load float, float* %12, align 4
  %22 = fadd float %18, %21
  store float %22, float* %12, align 4
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %exitcond11 = icmp ne i64 %indvars.iv.next10, 256
  br i1 %exitcond11, label %13, label %23

23:                                               ; preds = %13
  %24 = load float, float* %11, align 4
  %25 = load float, float* %12, align 4
  %26 = fadd float %24, %25
  %27 = getelementptr inbounds float, float* %5, i64 %indvars.iv12
  %28 = load float, float* %27, align 4
  %29 = fadd float %26, %28
  %30 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 %indvars.iv15, i64 %indvars.iv12
  store float %29, float* %30, align 4
  %indvars.iv.next13 = add nuw nsw i64 %indvars.iv12, 1
  %exitcond14 = icmp ne i64 %indvars.iv.next13, 256
  br i1 %exitcond14, label %10, label %31

31:                                               ; preds = %23
  %indvars.iv.next16 = add nuw nsw i64 %indvars.iv15, 1
  %exitcond17 = icmp ne i64 %indvars.iv.next16, 4
  br i1 %exitcond17, label %.preheader4, label %.preheader

32:                                               ; preds = %.preheader, %32
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %32 ]
  %33 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  %34 = load float, float* %33, align 4
  %35 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 2, i64 %indvars.iv
  %36 = load float, float* %35, align 4
  %37 = fadd float %36, 1.000000e+00
  %38 = fmul float %34, %37
  %39 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 0, i64 %indvars.iv
  %40 = load float, float* %39, align 4
  %41 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 1, i64 %indvars.iv
  %42 = load float, float* %41, align 4
  %43 = fmul float %40, %42
  %44 = fadd float %38, %43
  store float %44, float* %33, align 4
  %45 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 3, i64 %indvars.iv
  %46 = load float, float* %45, align 4
  %47 = fmul float %44, %46
  %48 = getelementptr inbounds float, float* %2, i64 %indvars.iv
  store float %47, float* %48, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %32, label %49

49:                                               ; preds = %32
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
