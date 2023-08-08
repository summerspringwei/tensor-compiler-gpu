#pragma once

namespace souffle {
namespace gpt2 {

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
    kChunkK = 2,
    kStage = 3,
    kBlockRowWarps = 2,
    kBlockColWarps = 2,
    kWarpSize = 32,
};


namespace FeedForwardFC1Params{
    enum FC1Params {
        // M=5120, N=384, K=1280
        // ctm_m=64, cta_n=128, cta_k=32
        KGEMMFFM = 5120,
        KGEMMFFN = 384,
        KGEMMFFK = 1280,

        kWarpRowTiles = 4,
        kWarpColTiles = 3,
        
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

        kWarpRowTiles = 4,
        kWarpColTiles = 2,
        kMTiles = 2,
        kNTiles = 1,
        
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kGridBlocks = (KGEMMFFM / kBlockRowTiles / kWmmaM / kMTiles) * 
            (KGEMMFFN / kBlockColTiles / kWmmaN / kNTiles),
    };

} // namespace FeedForwardFC1LimitedBlocksParams

namespace FeedForwardFC2Params{
    enum FC2Params {
        kGemmK6BlockRowTiles = 4,
        kGemmK6BlockColTiles = 4,
        kGemmK6BlockSliceKTiles = 4,
    };

} // namespace FeedForwardFC2Params

} // namespace gpt2
} // namespace souffle
