#pragma once

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
}

struct GPTLLargeFeedForwardFC1Params {
    
}

}