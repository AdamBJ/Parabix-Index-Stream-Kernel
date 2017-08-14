/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include "idx_stream_kernel.h"
#include <kernels/kernel_builder.h>  // for IDISA_Builder
#include <llvm/Support/raw_ostream.h>
#include <iostream>

using namespace llvm;
namespace kernel {
	void IdxStreamKernel::generateFinalBlockMethod(const std::unique_ptr<KernelBuilder> & kb, Value * remainingItems) {
	    for (unsigned j = 0; j < mStreamCount; ++j) {	
	//kb->CallPrintInt("numb remaining items in final: ", remainingItems);
			// Call generateDoBlockMethod
			CreateDoBlockMethodCall(kb);
	
			// Write the final idx pack to output, regardless of whether it's full or not
			Type * iPackTy = kb->getIntNTy(mPackSize);
			Type * iPackTyPtr = iPackTy->getPointerTo();
			Value * idxBitsAccumulator = kb->getScalarField("idxBitsAccumulator" + std::to_string(j));
			writeIdxStrmToOutput(iPackTyPtr, idxBitsAccumulator, kb, j);

			kb->setScalarField("idxBitsAccumulator" + std::to_string(j), kb->getSize(0));
			kb->setScalarField("numAccumulatedIdxBits" + std::to_string(j), kb->getSize(0));
		}
	}

	void IdxStreamKernel::generateDoBlockMethod(const std::unique_ptr<KernelBuilder> & kb) {
		for (unsigned j = 0; j < mStreamCount; ++j) {		
			BasicBlock * entryBlock = kb->GetInsertBlock();   
			BasicBlock * writeToOutputBlock = kb->CreateBasicBlock("writeToOutputBlock");
			BasicBlock * returnBlock = kb->CreateBasicBlock("returnBlock");
	
			Type * iPackTy = kb->getIntNTy(mPackSize);
			Type * iPackTyPtr = iPackTy->getPointerTo();		
			unsigned const blockWidth = kb->getBitBlockWidth();
			unsigned const packsPerBlock = blockWidth/mPackSize;
			Value * idxBitsAccumulator = kb->getScalarField("idxBitsAccumulator" + std::to_string(j));
			Value * numAccumulatedIdxBits = kb->getScalarField("numAccumulatedIdxBits" + std::to_string(j));
	//kb->CallPrintInt("numAccumulatedIdxBits_phi", numAccumulatedIdxBits); //TODO remove
	
			// Determine which mPackSize segments of this block contain at least 
			// a single set bit
			Value * blk = kb->loadInputStreamBlock("bitStreams", kb->getInt32(j));
		kb->CallPrintRegister("blk", blk); //TODO remove
			
			Value * hasBit = kb->simd_ugt(mPackSize, blk, kb->allZeroes()); 
			Value * blkIdxBits = kb->CreateZExtOrTrunc(kb->hsimd_signmask(mPackSize, hasBit), 
				iPackTy); 
	//kb->CallPrintInt("blkIdxBits", blkIdxBits); //TODO remove
				
			// We've generated blockWidth/mPackSize index bits (one for each mPackSized 
			// segment of the current block). Append these bits to idxBitsAccumulator stream.
			// Remember that streams should grow from right to left -- the more recent the addition
			// to the stream, the farther left in the stream it should be. Final block blkIdxBits 
			// should be leftmost bits of idxBitsAccumulator		
			// Value * relBlockNo = kb->CreateUDiv(numAccumulatedIdxBits, kb->getSize(packsPerBlock));
			////kb->CallPrintInt("relBlockNo", relBlockNo); //TODO remove
			
			// Value * shiftAmount = kb->CreateMul(relBlockNo, kb->getSize(packsPerBlock));
			//kb->CallPrintInt("shiftAmount", shiftAmount); //TODO remove
			//kb->CallPrintInt("numAccumulatedIdxBits", numAccumulatedIdxBits); //TODO remove
			
			// Preserve the existing bits in the accumulator by shifting the new index bits left before ORing them into the accumulator
			idxBitsAccumulator = kb->CreateOr(idxBitsAccumulator, kb->CreateShl(blkIdxBits, numAccumulatedIdxBits)); // shiftAmount			
			numAccumulatedIdxBits = kb->CreateAdd(numAccumulatedIdxBits, kb->getSize(packsPerBlock));

			// If we've accumulated a full pack of index bits (mPackSize bits), write the index bits to output
			Value * haveFullPack = kb->CreateICmpEQ(numAccumulatedIdxBits, kb->getSize(mPackSize));
			kb->CreateCondBr(haveFullPack, writeToOutputBlock, returnBlock);
		
			kb->SetInsertPoint(writeToOutputBlock);
			writeIdxStrmToOutput(iPackTyPtr, idxBitsAccumulator, kb, j);
			Value * resetIdxBits = kb->getSize(0);
			Value * resetnumAccumulatedIdxBits = kb->getSize(0);		
			kb->CreateBr(returnBlock);
				
			kb->SetInsertPoint(returnBlock);		
			PHINode * idxBitsAccumulator_phi = kb->CreatePHI(kb->getSizeTy(), 2);
			PHINode * numAccumulatedIdxBits_phi = kb->CreatePHI(kb->getSizeTy(), 2);
			idxBitsAccumulator_phi->addIncoming(resetIdxBits, writeToOutputBlock);
			numAccumulatedIdxBits_phi->addIncoming(resetnumAccumulatedIdxBits, writeToOutputBlock);
			idxBitsAccumulator_phi->addIncoming(idxBitsAccumulator, entryBlock);
			numAccumulatedIdxBits_phi->addIncoming(numAccumulatedIdxBits, entryBlock);
			kb->setScalarField("idxBitsAccumulator" + std::to_string(j), idxBitsAccumulator_phi);
			kb->setScalarField("numAccumulatedIdxBits" + std::to_string(j), numAccumulatedIdxBits_phi);
			
	//kb->CallPrintInt("idxBitsAccumulator_phi", idxBitsAccumulator_phi); //TODO remove
	//kb->CallPrintInt("numAccumulatedIdxBits_phi", numAccumulatedIdxBits_phi); //TODO remove
	    }
	}

	IdxStreamKernel::IdxStreamKernel(const std::unique_ptr<kernel::KernelBuilder> & kb, unsigned int packSize, unsigned int numInputStreams)
	: BlockOrientedKernel("IdxStreamSet"
	, {Binding{kb->getStreamSetTy(numInputStreams, 1), "bitStreams"}}
	// All input streams in IdxStreamSet have same num blocks to process, same fw
	, {Binding{kb->getStreamSetTy(numInputStreams, 1), "idxStreams", FixedRatio(1, packSize)}} 
	, {}
	, {}
	, {})
	, mPackSize(packSize)
	, mStreamCount(numInputStreams) {
        // TODO assert packsize%blocksize == 0?
        for (unsigned i = 0; i < mStreamCount; i++) {
        	mInternalScalars.emplace_back(kb->getSizeTy(), "idxBitsAccumulator" + std::to_string(i));
        	mInternalScalars.emplace_back(kb->getSizeTy(), "numAccumulatedIdxBits" + std::to_string(i));
        	mInternalScalars.emplace_back(kb->getSizeTy(), "numberOfWrites" + std::to_string(i));
        }
	}

    void IdxStreamKernel::writeIdxStrmToOutput(Type * iPackTyPtr, Value * idxBitsAccumulator, const std::unique_ptr<KernelBuilder> & kb,
    	unsigned streamNum) {
		Value * idxStreamBlockPtr = kb->CreateBitCast(kb->getOutputStreamBlockPtr("idxStreams", kb->getInt32(streamNum)),
			iPackTyPtr);
        Value * numberWrites = kb->getScalarField("numberOfWrites" + std::to_string(streamNum));
        // field width of idxStream is 1 bit, but we've casted idxSteamBlockPtr to iPackTyPtr (usually 64 bit ptr). Adding 1 to offset  of idxStreamBlockPtr adds 64bits, not 1 bit!
        Value * idxStreamOutputPtr =  kb->CreateGEP(idxStreamBlockPtr, numberWrites); 
        kb->setScalarField("numberOfWrites" + std::to_string(streamNum), kb->CreateAdd(numberWrites, kb->getSize(1)));
        kb->CreateStore(idxBitsAccumulator, idxStreamOutputPtr); //TODO storing a 64-bit quantity here... that first 5 bits are 0. That's why we have 0 in the output?
		//kb->CallPrintInt("just wrote to output: ", idxBitsAccumulator); //TODO remove
    }
}
