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
        kWarpColTiles = 2,
        
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kGridBlocks = KGEMMFFM / kBlockRowTiles / kWmmaM * KGEMMFFN / kBlockColTiles / kWmmaN,
    };

} // namespace Large
} // namespace gpt2
} // namespace souffle
