#pragma once
#include <cuda_fp16.h>

namespace souffle {
namespace gpt2 {

enum GPUParams {
    kNumberSM = 82,
    kElementwiseBlockThreads = 32*8,
};

enum GPT2LargeParams {
    kBatchSize = 1,
    kSeqLength = 384,
    kHeadNum = 20,
    kHeadSize = 64,
    kLayerNum = 36,
    kHiddenSize = 4,
    kHiddenDim = kHeadNum * kHeadSize,
};


enum GPTGEMMParams {
    kWmmaM = 16,
    kWmmaN = 16,
    kWmmaK = 16,
    kInputSkew = 8,
    kAccSkew = 8,
    kStage = 3,
    kWarpSize = 32,
};

namespace AttnQKVParams {
    enum QKVParams {
        kM = 1280,
        kN = 384,
        kK = 1280*3,

        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        kChunkK = 2,
        kGemmK1WarpRowTiles = 2,
        kGemmK1WarpColTiles = 3,
        
        kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks = kM / kBlockRowTiles / kWmmaM * kN / kBlockColTiles / kWmmaN,
        kSharedMemory = kStage * (3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
            (kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew))) * sizeof(half),
    };
}// namespace AttnQKVParams

namespace AttnQueryKeyParams {
    enum QueryKeyParams {
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        kChunkK = 4,
        kGemmK2WarpRowTiles = 4,
        kGemmK2WarpColTiles = 4,
        kGemmK2BatchedNum = kHeadNum,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks =
            (kSeqLength / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) *
            (kSeqLength / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) *
            kGemmK2BatchedNum,
        kSharedMemory = ((kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM) * (kChunkK * kWmmaK + kInputSkew) + (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN) * (kChunkK * kWmmaK + kInputSkew)) * sizeof(half),
    };
}

namespace AttnQueryKeyParamsLimitedBlocks {
    enum QueryKeyParams {
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        kChunkK = 4,
        kGemmK2WarpRowTiles = 4,
        kGemmK2WarpColTiles = 4,
        kGemmK2BatchedNum = kHeadNum,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kBatchTiles = 2,
        kGridBlocks =
            (kSeqLength / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) *
            (kSeqLength / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) *
            kGemmK2BatchedNum / kBatchTiles,
        kSharedMemory = ((kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM) * (kChunkK * kWmmaK + kInputSkew) + (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN) * (kChunkK * kWmmaK + kInputSkew)) * sizeof(half),
    };
}

namespace AttnValueParams {
    enum AttnValueParams {
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        kChunkK = 4,
        kGemmK3WarpRowTiles = 2,
        kGemmK3WarpColTiles = 2,
        kGemmK3BatchedNum = kHeadNum,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks =
            (kHeadSize / (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM)) *
            (kSeqLength / (kBlockColWarps * kGemmK3WarpColTiles * kWmmaN)) *
            kGemmK3BatchedNum,
        kSharedMemory =
                (kStage *
                (kChunkK * kWmmaK *
                    (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM + kInputSkew) +
                kBlockColWarps * kGemmK3WarpColTiles * kWmmaN *
                    (kChunkK * kWmmaK + kInputSkew))) *
                sizeof(half),
    };
}

namespace AttnFcParams {
    // (384, 1280) * (1280, 1280) -> (384, 1280)
    // Each block computes(64, 64)，we need 6*20 blocks
    enum AttnFcParams {
        kGemmK4WarpRowTiles = 2,
        kGemmK4WarpColTiles = 3,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        kChunkK = 4,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks =
            (kHeadNum * kHeadSize / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
            (kSeqLength / (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN)),
        kSharedMemory =
        (kStage *
            (kChunkK * kWmmaK *
                (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
            kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
                (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half),
    };
}

namespace FeedForwardFC1Params{
    enum FC1Params {
        // M=5120, N=384, K=1280
        // ctm_m=64, cta_n=128, cta_k=32
        KGEMMFFM = 5120,
        KGEMMFFN = 384,
        KGEMMFFK = 1280,

        kWarpRowTiles = 4,
        kWarpColTiles = 3,
        kChunkK = 2,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kGridBlocks = KGEMMFFM / kBlockRowTiles / kWmmaM * KGEMMFFN / kBlockColTiles / kWmmaN,
    };

} // namespace FeedForwardFC1Params

namespace FeedForwardFC1LimitedBlocksParams{
    enum FC1Params {
        // M=5120, N=384, K=1280
        // ctm_m=64, cta_n=128, cta_k=32
        KGEMMFFM = 5120,
        KGEMMFFN = 384,
        KGEMMFFK = 1280,
        kGEMMFFB = 1,

        kWarpRowTiles = 4,
        kWarpColTiles = 3,
        kMTiles = 2,
        kNTiles = 1,
        kChunkK = 2,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks = (KGEMMFFM / kBlockRowTiles / kWmmaM / kMTiles) * 
            (KGEMMFFN / kBlockColTiles / kWmmaN / kNTiles),
        
        kSharedMemory = (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps *
                   FeedForwardFC1LimitedBlocksParams::kBlockRowTiles * kWmmaM +
               kInputSkew) +
          kBlockColWarps * FeedForwardFC1LimitedBlocksParams::kBlockColTiles *
              kWmmaN * (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half),
    };
} // namespace FeedForwardFC1LimitedBlocksParams


// weight(k, n) * input(m, k) = output(m, n):
//  (5120, 1280) * (5120, 384) -> (1280, 384)
namespace FeedForwardFC2Params{
    enum FC2Params {
        KGEMMFFM = 1280,
        KGEMMFFN = 384,
        KGEMMFFK = 5120,

        kGemmK6BlockRowTiles = 8,
        kGemmK6BlockColTiles = 4,
        // May set kGEEMK6BlockSliceKTiles = 5 for A100
        kGemmK6BlockSliceKTiles = 4,
        kChunkK = 2,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,

        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks = (KGEMMFFM / (kGemmK6BlockRowTiles * kWmmaM)) *
        (KGEMMFFN / (kGemmK6BlockColTiles * kWmmaN)),

        kSharedMemory =
        (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                       (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                   kGemmK6BlockColTiles * kWmmaN *
                       (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
        sizeof(half),
    };
} // namespace FeedForwardFC2Params

} // namespace gpt2
} // namespace souffle
