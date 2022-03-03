; ModuleID = 'lstm_fusion_interchange.ll'
source_filename = "lstm_fusion_interchange.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@num_hidden = dso_local constant i32 256, align 4
@num_gate = dso_local constant i32 4, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @lstm_cell_fusion_interchange_fusion(float* %0, float* %1, float* %2, [256 x [256 x float]]* %3, [256 x [256 x float]]* %4, float* %5) #0 {
  %7 = alloca [4 x [256 x float]], align 16
  %8 = alloca [4 x [256 x float]], align 16
  %9 = alloca [4 x [256 x float]], align 16
  br label %.split

.split:                                           ; preds = %6
  br label %.preheader

.preheader:                                       ; preds = %.split, %34
  %indvars.iv9 = phi i64 [ 0, %.split ], [ %indvars.iv.next10, %34 ]
  br label %10

10:                                               ; preds = %.preheader, %26
  %indvars.iv6 = phi i64 [ 0, %.preheader ], [ %indvars.iv.next7, %26 ]
  %11 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %7, i64 0, i64 %indvars.iv6, i64 %indvars.iv9
  store float 0.000000e+00, float* %11, align 4
  %12 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %8, i64 0, i64 %indvars.iv6, i64 %indvars.iv9
  store float 0.000000e+00, float* %12, align 4
  br label %13

13:                                               ; preds = %10, %13
  %indvars.iv = phi i64 [ 0, %10 ], [ %indvars.iv.next, %13 ]
  %14 = getelementptr inbounds float, float* %0, i64 %indvars.iv
  %15 = load float, float* %14, align 4
  %16 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* %3, i64 %indvars.iv6, i64 %indvars.iv9, i64 %indvars.iv
  %17 = load float, float* %16, align 4
  %18 = fmul float %15, %17
  %19 = load float, float* %11, align 4
  %20 = fadd float %19, %18
  store float %20, float* %11, align 4
  %21 = getelementptr inbounds float, float* %2, i64 %indvars.iv
  %22 = load float, float* %21, align 4
  %23 = fmul float %17, %22
  %24 = load float, float* %12, align 4
  %25 = fadd float %24, %23
  store float %25, float* %12, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %13, label %26

26:                                               ; preds = %13
  %27 = load float, float* %11, align 4
  %28 = load float, float* %12, align 4
  %29 = fadd float %27, %28
  %30 = getelementptr inbounds float, float* %5, i64 %indvars.iv9
  %31 = load float, float* %30, align 4
  %32 = fadd float %29, %31
  %33 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 %indvars.iv6, i64 %indvars.iv9
  store float %32, float* %33, align 4
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %exitcond8 = icmp ne i64 %indvars.iv.next7, 4
  br i1 %exitcond8, label %10, label %34

34:                                               ; preds = %26
  %35 = getelementptr inbounds float, float* %1, i64 %indvars.iv9
  %36 = load float, float* %35, align 4
  %37 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 2, i64 %indvars.iv9
  %38 = load float, float* %37, align 4
  %39 = fadd float %38, 1.000000e+00
  %40 = fmul float %36, %39
  %41 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 0, i64 %indvars.iv9
  %42 = load float, float* %41, align 4
  %43 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 1, i64 %indvars.iv9
  %44 = load float, float* %43, align 4
  %45 = fmul float %42, %44
  %46 = fadd float %40, %45
  store float %46, float* %35, align 4
  %47 = getelementptr inbounds [4 x [256 x float]], [4 x [256 x float]]* %9, i64 0, i64 3, i64 %indvars.iv9
  %48 = load float, float* %47, align 4
  %49 = fmul float %46, %48
  %50 = getelementptr inbounds float, float* %2, i64 %indvars.iv9
  store float %49, float* %50, align 4
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %exitcond11 = icmp ne i64 %indvars.iv.next10, 256
  br i1 %exitcond11, label %.preheader, label %51

51:                                               ; preds = %34
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
