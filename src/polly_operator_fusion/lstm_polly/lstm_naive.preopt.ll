; ModuleID = 'lstm_naive.ll'
source_filename = "lstm_naive.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@num_hidden = dso_local constant i32 256, align 4
@num_gate = dso_local constant i32 4, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @lstm_cell_naive(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
  %7 = alloca [4 x [256 x float]], align 16
  %8 = alloca [4 x [256 x float]], align 16
  %9 = alloca [4 x [256 x float]], align 16
  br label %.split

.split:                                           ; preds = %6
  br label %.preheader11

.preheader11:                                     ; preds = %.split, %42
  %indvars.iv38 = phi i64 [ 0, %.split ], [ %indvars.iv.next39, %42 ]
  br label %10

.preheader8:                                      ; preds = %42
  br label %43

.preheader10:                                     ; preds = %20
  br label %21

10:                                               ; preds = %.preheader11, %20
  %indvars.iv26 = phi i64 [ 0, %.preheader11 ], [ %indvars.iv.next27, %20 ]
  %11 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %7, i64 0, i64 %indvars.iv38, i64 %indvars.iv26
  store float 0.000000e+00, float* %11, align 4
  br label %12

12:                                               ; preds = %10, %12
  %indvars.iv23 = phi i64 [ 0, %10 ], [ %indvars.iv.next24, %12 ]
  %13 = getelementptr inbounds float, float* %0, i64 %indvars.iv23
  %14 = load float, float* %13, align 4
  %15 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %3, i64 %indvars.iv38, i64 %indvars.iv26, i64 %indvars.iv23
  %16 = load float, float* %15, align 4
  %17 = fmul float %14, %16
  %18 = load float, float* %11, align 4
  %19 = fadd float %18, %17
  store float %19, float* %11, align 4
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exitcond25 = icmp ne i64 %indvars.iv.next24, 256
  br i1 %exitcond25, label %12, label %20

20:                                               ; preds = %12
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  %exitcond28 = icmp ne i64 %indvars.iv.next27, 256
  br i1 %exitcond28, label %10, label %.preheader10

.preheader9:                                      ; preds = %31
  br label %32

21:                                               ; preds = %.preheader10, %31
  %indvars.iv32 = phi i64 [ 0, %.preheader10 ], [ %indvars.iv.next33, %31 ]
  %22 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %8, i64 0, i64 %indvars.iv38, i64 %indvars.iv32
  store float 0.000000e+00, float* %22, align 4
  br label %23

23:                                               ; preds = %21, %23
  %indvars.iv29 = phi i64 [ 0, %21 ], [ %indvars.iv.next30, %23 ]
  %24 = getelementptr inbounds float, float* %2, i64 %indvars.iv29
  %25 = load float, float* %24, align 4
  %26 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %4, i64 %indvars.iv38, i64 %indvars.iv32, i64 %indvars.iv29
  %27 = load float, float* %26, align 4
  %28 = fmul float %25, %27
  %29 = load float, float* %22, align 4
  %30 = fadd float %29, %28
  store float %30, float* %22, align 4
  %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1
  %exitcond31 = icmp ne i64 %indvars.iv.next30, 256
  br i1 %exitcond31, label %23, label %31

31:                                               ; preds = %23
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond34 = icmp ne i64 %indvars.iv.next33, 256
  br i1 %exitcond34, label %21, label %.preheader9

32:                                               ; preds = %.preheader9, %32
  %indvars.iv35 = phi i64 [ 0, %.preheader9 ], [ %indvars.iv.next36, %32 ]
  %33 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %7, i64 0, i64 %indvars.iv38, i64 %indvars.iv35
  %34 = load float, float* %33, align 4
  %35 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %8, i64 0, i64 %indvars.iv38, i64 %indvars.iv35
  %36 = load float, float* %35, align 4
  %37 = fadd float %34, %36
  %38 = getelementptr inbounds float, float* %5, i64 %indvars.iv35
  %39 = load float, float* %38, align 4
  %40 = fadd float %37, %39
  %41 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 %indvars.iv38, i64 %indvars.iv35
  store float %40, float* %41, align 4
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond37 = icmp ne i64 %indvars.iv.next36, 256
  br i1 %exitcond37, label %32, label %42

42:                                               ; preds = %32
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond40 = icmp ne i64 %indvars.iv.next39, 4
  br i1 %exitcond40, label %.preheader11, label %.preheader8

.preheader:                                       ; preds = %43
  br label %56

43:                                               ; preds = %.preheader8, %43
  %indvars.iv20 = phi i64 [ 0, %.preheader8 ], [ %indvars.iv.next21, %43 ]
  %44 = getelementptr inbounds float, float* %1, i64 %indvars.iv20
  %45 = load float, float* %44, align 4
  %46 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 2, i64 %indvars.iv20
  %47 = load float, float* %46, align 4
  %48 = fadd float %47, 1.000000e+00
  %49 = fmul float %45, %48
  %50 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 0, i64 %indvars.iv20
  %51 = load float, float* %50, align 4
  %52 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 1, i64 %indvars.iv20
  %53 = load float, float* %52, align 4
  %54 = fmul float %51, %53
  %55 = fadd float %49, %54
  store float %55, float* %44, align 4
  %indvars.iv.next21 = add nuw nsw i64 %indvars.iv20, 1
  %exitcond22 = icmp ne i64 %indvars.iv.next21, 256
  br i1 %exitcond22, label %43, label %.preheader

56:                                               ; preds = %.preheader, %56
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %56 ]
  %57 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  %58 = load float, float* %57, align 4
  %59 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 3, i64 %indvars.iv
  %60 = load float, float* %59, align 4
  %61 = fmul float %58, %60
  %62 = getelementptr inbounds float, float* %2, i64 %indvars.iv
  store float %61, float* %62, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %56, label %63

63:                                               ; preds = %56
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
