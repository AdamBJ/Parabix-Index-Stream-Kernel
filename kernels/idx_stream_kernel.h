/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef IDX_STREAM_H
#define IDX_STREAM_H

#include "kernel.h"  // for KernelBuilder
namespace IDISA { class IDISA_Builder; }  // lines 14-14 TODO remove this?
namespace llvm { class Value; } //TODO what are these for?

namespace kernel {

class IdxStreamKernel : public BlockOrientedKernel {
public:
    IdxStreamKernel(const std::unique_ptr<KernelBuilder> & builder, unsigned int packSize = 64, unsigned int numInputStreams = 1);
protected:
    void generateDoBlockMethod(const std::unique_ptr<KernelBuilder> & idb) override;
    void generateFinalBlockMethod(const std::unique_ptr<KernelBuilder> & idb, llvm::Value * remainingItems) override;
    void writeIdxStrmToOutput(llvm::Type * iPackPtrTy, llvm::Value * idxBits, const std::unique_ptr<KernelBuilder> & idb, unsigned streamNum);
private:    
    const unsigned mPackSize;
    const unsigned mStreamCount;
};

}
#endif
