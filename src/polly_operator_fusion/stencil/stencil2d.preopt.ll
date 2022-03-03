; ModuleID = 'stencil/stencil2d.ll'
source_filename = "stencil/stencil2d.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@L = dso_local constant i32 10, align 4
@M = dso_local constant i32 10, align 4
@N = dso_local constant i32 10, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @stencil2d([10 x float]* %0) #0 {
  br label %.split

.split:                                           ; preds = %1
  br label %.preheader3

.preheader3:                                      ; preds = %.split, %18
  %.016 = phi i32 [ 0, %.split ], [ %19, %18 ]
  br label %.preheader

.loopexit:                                        ; preds = %2
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 9
  br i1 %exitcond9, label %.preheader, label %18

.preheader:                                       ; preds = %.preheader3, %.loopexit
  %indvars.iv7 = phi i64 [ 1, %.preheader3 ], [ %indvars.iv.next8, %.loopexit ]
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  br label %2

2:                                                ; preds = %.preheader, %2
  %indvars.iv = phi i64 [ 1, %.preheader ], [ %indvars.iv.next, %2 ]
  %3 = getelementptr inbounds [10 x float], [10 x float]* %0, i64 %indvars.iv.next8, i64 %indvars.iv
  %4 = load float, float* %3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %5 = getelementptr inbounds [10 x float], [10 x float]* %0, i64 %indvars.iv7, i64 %indvars.iv.next
  %6 = load float, float* %5, align 4
  %7 = fadd float %4, %6
  %8 = add nsw i64 %indvars.iv7, -1
  %9 = getelementptr inbounds [10 x float], [10 x float]* %0, i64 %8, i64 %indvars.iv
  %10 = load float, float* %9, align 4
  %11 = fadd float %7, %10
  %12 = add nsw i64 %indvars.iv, -1
  %13 = getelementptr inbounds [10 x float], [10 x float]* %0, i64 %indvars.iv7, i64 %12
  %14 = load float, float* %13, align 4
  %15 = fadd float %11, %14
  %16 = fmul float %15, 2.500000e-01
  %17 = getelementptr inbounds [10 x float], [10 x float]* %0, i64 %indvars.iv7, i64 %indvars.iv
  store float %16, float* %17, align 4
  %exitcond = icmp ne i64 %indvars.iv.next, 9
  br i1 %exitcond, label %2, label %.loopexit

18:                                               ; preds = %.loopexit
  %19 = add nuw nsw i32 %.016, 1
  %exitcond10 = icmp ne i32 %19, 10
  br i1 %exitcond10, label %.preheader3, label %20

20:                                               ; preds = %18
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca [10 x [10 x float]], align 16
  br label %.split

.split:                                           ; preds = %0
  %2 = getelementptr inbounds [10 x [10 x float]], [10 x [10 x float]]* %1, i64 0, i64 0
  call void @stencil2d([10 x float]* nonnull %2)
  ret i32 0
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
