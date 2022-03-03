; ModuleID = 'maxpool/maxpool.ll'
source_filename = "maxpool/maxpool.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@n = dso_local constant i32 1, align 4
@c = dso_local constant i32 16, align 4
@h = dso_local constant i32 32, align 4
@w = dso_local constant i32 32, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @maxpool([16 x [32 x [32 x float]]]* %0, [16 x [32 x [32 x float]]]* %1) #0 {
  br label %.split

.split:                                           ; preds = %2
  br label %.preheader12

.preheader12:                                     ; preds = %.split, %23
  %indvars.iv27 = phi i64 [ 0, %.split ], [ %indvars.iv.next28, %23 ]
  br label %.preheader11

.preheader11:                                     ; preds = %.preheader12, %22
  %indvars.iv24 = phi i64 [ 0, %.preheader12 ], [ %indvars.iv.next25, %22 ]
  br label %.preheader10

.preheader10:                                     ; preds = %.preheader11, %21
  %indvars.iv21 = phi i64 [ 0, %.preheader11 ], [ %indvars.iv.next22, %21 ]
  br label %.preheader9

.preheader9:                                      ; preds = %.preheader10, %20
  %indvars.iv = phi i64 [ 0, %.preheader10 ], [ %indvars.iv.next, %20 ]
  br label %.preheader

.preheader:                                       ; preds = %.preheader9, %18
  %.0614 = phi i32 [ 0, %.preheader9 ], [ %19, %18 ]
  br label %3

3:                                                ; preds = %.preheader, %15
  %.0113 = phi i32 [ 0, %.preheader ], [ %17, %15 ]
  %4 = add nsw i64 %indvars.iv21, -1
  %5 = trunc i64 %4 to i32
  %6 = add nsw i32 %5, %.0614
  %7 = add nsw i64 %indvars.iv, -1
  %8 = trunc i64 %7 to i32
  %9 = add nsw i32 %8, %.0113
  %10 = or i32 %9, %6
  %11 = icmp ult i32 %10, 32
  br i1 %11, label %12, label %15

12:                                               ; preds = %3
  %13 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %0, i64 %indvars.iv27, i64 %indvars.iv24, i64 %indvars.iv21, i64 %indvars.iv
  %14 = load float, float* %13, align 4
  br label %15

15:                                               ; preds = %12, %3
  %.0 = phi float [ %14, %12 ], [ -1.000000e+04, %3 ]
  %16 = getelementptr inbounds [16 x [32 x [32 x float]]], [16 x [32 x [32 x float]]]* %1, i64 %indvars.iv27, i64 %indvars.iv24, i64 %indvars.iv21, i64 %indvars.iv
  store float %.0, float* %16, align 4
  %17 = add nuw nsw i32 %.0113, 1
  %exitcond = icmp ne i32 %17, 3
  br i1 %exitcond, label %3, label %18

18:                                               ; preds = %15
  %19 = add nuw nsw i32 %.0614, 1
  %exitcond19 = icmp ne i32 %19, 3
  br i1 %exitcond19, label %.preheader, label %20

20:                                               ; preds = %18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond20 = icmp ne i64 %indvars.iv.next, 32
  br i1 %exitcond20, label %.preheader9, label %21

21:                                               ; preds = %20
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp ne i64 %indvars.iv.next22, 32
  br i1 %exitcond23, label %.preheader10, label %22

22:                                               ; preds = %21
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %exitcond26 = icmp ne i64 %indvars.iv.next25, 16
  br i1 %exitcond26, label %.preheader11, label %23

23:                                               ; preds = %22
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1
  br i1 false, label %.preheader12, label %24

24:                                               ; preds = %23
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 "}
