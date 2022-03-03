; ModuleID = 'softmax.ll'
source_filename = "softmax.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@n = dso_local constant i32 256, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @softmax(float* %0, float* %1) #0 {
  br label %.split

.split:                                           ; preds = %2
  br label %3

.preheader:                                       ; preds = %3
  %.lcssa = phi float [ %7, %3 ]
  br label %8

3:                                                ; preds = %.split, %3
  %indvars.iv6 = phi i64 [ 0, %.split ], [ %indvars.iv.next7, %3 ]
  %.024 = phi float [ 0.000000e+00, %.split ], [ %7, %3 ]
  %4 = getelementptr inbounds float, float* %0, i64 %indvars.iv6
  %5 = load float, float* %4, align 4
  %6 = getelementptr inbounds float, float* %1, i64 %indvars.iv6
  store float %5, float* %6, align 4
  %7 = fadd float %.024, %5
  %indvars.iv.next7 = add nuw nsw i64 %indvars.iv6, 1
  %exitcond8 = icmp ne i64 %indvars.iv.next7, 256
  br i1 %exitcond8, label %3, label %.preheader

8:                                                ; preds = %.preheader, %8
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %8 ]
  %9 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  %10 = load float, float* %9, align 4
  %11 = fdiv float %10, %.lcssa
  store float %11, float* %9, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %8, label %12

12:                                               ; preds = %8
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
