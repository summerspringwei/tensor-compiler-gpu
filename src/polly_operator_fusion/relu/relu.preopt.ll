; ModuleID = 'relu.ll'
source_filename = "relu.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@n = dso_local constant i32 256, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @relu(float* %0, float* %1) #0 {
  br label %.split

.split:                                           ; preds = %2
  br label %3

3:                                                ; preds = %.split, %10
  %indvars.iv = phi i64 [ 0, %.split ], [ %indvars.iv.next, %10 ]
  %4 = getelementptr inbounds float, float* %0, i64 %indvars.iv
  %5 = load float, float* %4, align 4
  %6 = fcmp ogt float %5, 0.000000e+00
  %7 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  br i1 %6, label %8, label %9

8:                                                ; preds = %3
  br label %10

9:                                                ; preds = %3
  br label %10

10:                                               ; preds = %8, %9
  %storemerge = phi float [ 0.000000e+00, %9 ], [ %5, %8 ]
  store float %storemerge, float* %7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %3, label %11

11:                                               ; preds = %10
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
