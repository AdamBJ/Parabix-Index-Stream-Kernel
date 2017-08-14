/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include "idx_stream_kernel.h"
#include <kernels/kernel_builder.h>  // for IDISA_Builder
#include <llvm/Support/raw_ostream.h>
#include <iostream>
//TODO getting ...11, first 3 bits are meaningless. need 11000, or 00011000. Might just be issue with stdout_kernel

using namespace llvm;
namespace kernel {
	void IdxStreamKernel::generateFinalBlockMethod(const std::unique_ptr<KernelBuilder> & iBuilder, Value * remainingItems) {
	    for (unsigned j = 0; j < mStreamCount; ++j) {	
			iBuilder->CallPrintInt("numb rem items in final: ", remainingItems);
			// call generateDoBlockMethod
			CreateDoBlockMethodCall(iBuilder);
	
			// write the idx pack to output
			Type * iPackTy = iBuilder->getIntNTy(mPackSize);
			Type * iPackPtrTy = iPackTy->getPointerTo();
			Value * idxBitsAccumulator = iBuilder->getScalarField("idxBitsAccumulator" + std::to_string(j));
	
			writeIdxStrmToOutput(iPackPtrTy, idxBitsAccumulator, iBuilder, j);

			iBuilder->setScalarField("idxBitsAccumulator" + std::to_string(j), iBuilder->getSize(0));
			iBuilder->setScalarField("numIdxBits" + std::to_string(j), iBuilder->getSize(0));
		}
	}

	void IdxStreamKernel::generateDoBlockMethod(const std::unique_ptr<KernelBuilder> & iBuilder) {
		for (unsigned j = 0; j < mStreamCount; ++j) {		
			BasicBlock * entryBlock = iBuilder->GetInsertBlock();   
			BasicBlock * writeToOutputBlock = iBuilder->CreateBasicBlock("writeToOutputBlock");
			BasicBlock * returnBlock = iBuilder->CreateBasicBlock("returnBlock");
	
			Type * iPackTy = iBuilder->getIntNTy(mPackSize);
			Type * iPackPtrTy = iPackTy->getPointerTo();		
			unsigned const blockWidth = iBuilder->getBitBlockWidth();
			unsigned const packsPerBlock = blockWidth/mPackSize;
			Value * idxBitsAccumulator = iBuilder->getScalarField("idxBitsAccumulator" + std::to_string(j));
			Value * numIdxBits = iBuilder->getScalarField("numIdxBits" + std::to_string(j));
			iBuilder->CallPrintInt("numIdxBits_phi", numIdxBits); //TODO remove
	
			// Determine which mPackSize segments of this block contain at least 
			// a single 1 bit
			Value * blk = iBuilder->loadInputStreamBlock("bitStreams", iBuilder->getInt32(j));
			Value * hasBit = iBuilder->simd_ugt(mPackSize, blk, iBuilder->allZeroes()); 
			Value * blkIdxBits = iBuilder->CreateZExtOrTrunc(iBuilder->hsimd_signmask(mPackSize, hasBit), 
				iPackTy); 
					
			// We've generated blockWidth/mPackSize index bits (one for each mPackSized 
			// segment of the current block). Add these bits to idxBitsAccumulator stream.
			// Remember that streams should grow from right to left -- the more recent the addition
			// to the stream, the farther left in the stream it should be. Final block blkIdxBits 
			// should be leftmost bits of idxBitsAccumulator		
			Value * relBlockNo = iBuilder->CreateUDiv(numIdxBits, iBuilder->getSize(packsPerBlock));
			Value * shiftAmount = iBuilder->CreateMul(relBlockNo, iBuilder->getSize(packsPerBlock));
			idxBitsAccumulator = iBuilder->CreateOr(idxBitsAccumulator, iBuilder->CreateShl(blkIdxBits, shiftAmount));			
			numIdxBits = iBuilder->CreateAdd(numIdxBits, iBuilder->getSize(packsPerBlock));

			Value * haveFullPack = iBuilder->CreateICmpEQ(numIdxBits, iBuilder->getSize(mPackSize));
			iBuilder->CreateCondBr(haveFullPack, writeToOutputBlock, returnBlock);
		
			iBuilder->SetInsertPoint(writeToOutputBlock);
			writeIdxStrmToOutput(iPackPtrTy, idxBitsAccumulator, iBuilder, j);
			Value * resetIdxBits = iBuilder->getSize(0);
			Value * resetNumIdxBits = iBuilder->getSize(0);		
			iBuilder->CreateBr(returnBlock);
				
			iBuilder->SetInsertPoint(returnBlock);		
			PHINode * idxBitsAccumulator_phi = iBuilder->CreatePHI(iBuilder->getSizeTy(), 2);
			PHINode * numIdxBits_phi = iBuilder->CreatePHI(iBuilder->getSizeTy(), 2);
			idxBitsAccumulator_phi->addIncoming(resetIdxBits, writeToOutputBlock);
			numIdxBits_phi->addIncoming(resetNumIdxBits, writeToOutputBlock);
			idxBitsAccumulator_phi->addIncoming(idxBitsAccumulator, entryBlock);
			numIdxBits_phi->addIncoming(numIdxBits, entryBlock);
			
			iBuilder->CallPrintInt("idxBitsAccumulator_phi", idxBitsAccumulator_phi); //TODO remove
			iBuilder->CallPrintInt("numIdxBits_phi", numIdxBits_phi); //TODO remove
			
			iBuilder->setScalarField("idxBitsAccumulator" + std::to_string(j), idxBitsAccumulator_phi);
			iBuilder->setScalarField("numIdxBits" + std::to_string(j), numIdxBits_phi);
	    }
	}

	IdxStreamKernel::IdxStreamKernel(const std::unique_ptr<kernel::KernelBuilder> & iBuilder, unsigned int packSize, unsigned int numInputStreams)
	: BlockOrientedKernel("IdxStreamSet"
	, {Binding{iBuilder->getStreamSetTy(numInputStreams, 1), "bitStreams"}}
	// all input streams in IdxStreamSet have same num blocks to process, same fw
	, {Binding{iBuilder->getStreamSetTy(numInputStreams, 1), "idxStreams", FixedRatio(1, packSize)}} 
	, {}
	, {}
	, {/*{Binding{iBuilder->getSizeTy(), "idxBitsAccumulator0"}, Binding{iBuilder->getSizeTy(), "numIdxBits0"}, Binding{iBuilder->getSizeTy(), "numberOfWrites0"}}*/})
	, mPackSize(packSize)
	, mStreamCount(numInputStreams) {
        // TODO assert packsize%blocksize == 0?
        for (unsigned i = 0; i < mStreamCount; i++) {
        	mInternalScalars.emplace_back(iBuilder->getSizeTy(), "idxBitsAccumulator" + std::to_string(i));
        	mInternalScalars.emplace_back(iBuilder->getSizeTy(), "numIdxBits" + std::to_string(i));
        	mInternalScalars.emplace_back(iBuilder->getSizeTy(), "numberOfWrites" + std::to_string(i));
        }
	}

    void IdxStreamKernel::writeIdxStrmToOutput(Type * iPackPtrTy, Value * idxBitsAccumulator, const std::unique_ptr<KernelBuilder> & iBuilder,
    	unsigned streamNum) {
        Value * idxStreamBlockPtr = iBuilder->CreateBitCast(iBuilder->getOutputStreamBlockPtr("idxStreams",
	        iBuilder->getInt32(streamNum)), iPackPtrTy); //TODO member variable rather than calling each time?		
        Value * numberWrites = iBuilder->getScalarField("numberOfWrites" + std::to_string(streamNum));
        // field width of idxStream is 1 bit, but we've casted idxSteamBlockPtr to iPackPtrTy (usually 64 bit ptr). Adding 1 to offset  of idxStreamBlockPtr adds 64bits, not 1 bit!!
        Value * idxStreamOutputPtr =  iBuilder->CreateGEP(idxStreamBlockPtr, numberWrites); 
        iBuilder->setScalarField("numberOfWrites" + std::to_string(streamNum), iBuilder->CreateAdd(numberWrites, iBuilder->getSize(1)));
        iBuilder->CreateStore(idxBitsAccumulator, idxStreamOutputPtr); //TODO storing a 64-bit quantity here... that first 5 bits are 0. That's why we have 0 in the output?
        iBuilder->CallPrintInt("just wrote to output: ", idxBitsAccumulator); //TODO remove
    }
}
