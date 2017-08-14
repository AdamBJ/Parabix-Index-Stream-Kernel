/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include "idx_stream_kernel.h"
#include <kernels/kernel_builder.h>  // for KernelBuilder
#include <llvm/Support/raw_ostream.h>
#include <iostream>

using namespace llvm;
namespace kernel {
	void IdxStreamKernel::generateMultiBlockLogic(const std::unique_ptr<KernelBuilder> & kb) {
		/*
		Strategy:
		We know that we'll always have full blocks to process because of the guarentee provided by MultiBlockKernel.
		We'll either be processing multiple blocks or we'll be processing a final block that gets padded with zeroes. 
		Padding with zeroes doesn't affect final result so we treat the final block as a normal block.

		Roughly:
		while there are still blocks to process
			process a block
			check if the idxBitsAccumulator is full
				if full, write to output

		The bad news is that we don't know how many blocks we need to process at C++ runtime, so we have to write the while
		loop as an LLVM loop.

		*/	
		for (unsigned j = 0; j < mStreamCount; ++j) {	
			// Create variables
			BasicBlock * entry = kb->GetInsertBlock(); // TODO can this be removed?
			BasicBlock * checkLoopCond = kb->CreateBasicBlock("checkLoopCond");
			BasicBlock * processBlock = kb->CreateBasicBlock("processBlock");
			BasicBlock * writeToOutput = kb->CreateBasicBlock("writeToOutput");
			BasicBlock * terminate = kb->CreateBasicBlock("terminate");
			Constant * blockWidth = kb->getSize(kb->getBitBlockWidth());
			Constant * packsPerBlock = kb->getSize(kb->getBitBlockWidth()/mPackSize);
			Type * iPackTy = kb->getIntNTy(mPackSize);
			Type * iPackPtrTy = iPackTy->getPointerTo();
			Value * idxBitsAccumulator = kb->getScalarField("idxBitsAccumulator" + std::to_string(j));
			Value * numIdxBits = kb->getScalarField("numIdxBits" + std::to_string(j));

			// Extract function args
			Function::arg_iterator args = mCurrentMethod->arg_begin();
			args++; //self
			Value * itemsToDo = &*(args++);
			kb->CallPrintInt("itemsToDo", itemsToDo); //TODO remove
			kb->CallPrintInt("itemsToDo FOR STREAM NUMBER: ", kb->getSize(j)); //TODO remove
			Value * bitStreams = &*(args++); // streamSet with mStreamCount items
			Value * idxStreams = &*(args);
			Value * blocksToDo = kb->CreateUDiv(itemsToDo, blockWidth);
			// We treat final block as a normal block (padding with zeroes doesn't affect the result)
			blocksToDo = kb->CreateSelect(kb->CreateICmpEQ(blocksToDo, kb->getSize(0)), kb->getSize(1), blocksToDo);
			Value * blockNo = kb->getScalarField("blockNo" + std::to_string(j));
			kb->CreateBr(checkLoopCond);
            
			kb->SetInsertPoint(checkLoopCond);
			PHINode * idxBitsAccumulatorPhi = kb->CreatePHI(kb->getSizeTy(), 3);
			PHINode * numIdxBitsPhi = kb->CreatePHI(kb->getSizeTy(), 3);
			PHINode * blocksToDoPhi = kb->CreatePHI(kb->getSizeTy(), 3);
			PHINode * blockNoPhi = kb->CreatePHI(kb->getSizeTy(), 3);
			blocksToDoPhi->addIncoming(blocksToDo, entry);
			blockNoPhi->addIncoming(blockNo, entry);
			idxBitsAccumulatorPhi->addIncoming(idxBitsAccumulator, entry);
			numIdxBitsPhi->addIncoming(numIdxBits, entry);
			Value * haveRemBlocks = kb->CreateICmpUGT(blocksToDoPhi, kb->getSize(0));
			kb->CreateCondBr(haveRemBlocks, processBlock, terminate);

			kb->SetInsertPoint(processBlock);	
			// Determine which mPackSize segments of this block contain at least a single 1 bit
			// When dealing with a stream set containing more than one stream, each block we process is divided into numStreams
			// segments. We first find the block we want (blockNoPhi). Then find the segement belonging to the stream set we're interested
			// in (kb->getInt32(j))
			// Put another way, blockNoPhi moves us horizontally along the stream set until we reach the vertical slice that contains all the 
			// ith blocks. Then j gets us the block that belongs to the stream we want
			Value * blk = kb->CreateBlockAlignedLoad(kb->CreateGEP(bitStreams, {blockNoPhi, kb->getInt32(j)}));
			kb->CallPrintRegister("block", blk); //TODO remove
			kb->CallPrintInt("blockNoPhi", blockNoPhi); //TODO remove
			Value * hasBit = kb->simd_ugt(mPackSize, blk, kb->allZeroes());
			Value * blkIdxBits = kb->CreateZExtOrTrunc(kb->hsimd_signmask(mPackSize, hasBit), 
				iPackTy); 

			// We've generated blockWidth/mPackSize index bits (one for each mPackSized 
			// segment of the current block). OR these bits into idxBitsAccumulator stream.
			// Remember that streams should grow from right to left -- the more recent the addition
			// to the stream, the farther left in the stream it should be. Final block's blkIdxBits 
			// should be leftmost bits of idxBitsAccumulator		
			Value * relBlockNo = kb->CreateUDiv(numIdxBitsPhi, packsPerBlock);
			Value * shiftAmount = kb->CreateMul(relBlockNo, packsPerBlock);
			Value * idxBitsAccumulatorUpdated = kb->CreateOr(idxBitsAccumulatorPhi, kb->CreateShl(blkIdxBits, shiftAmount));		
			Value * numIdxBitsUpdated = kb->CreateAdd(numIdxBitsPhi, packsPerBlock);
			Value * blocksToDoUpdated = kb->CreateSub(blocksToDoPhi, kb->getSize(1));
			Value * blockNoUpdated = kb->CreateAdd(blockNoPhi, kb->getSize(1));
			kb->CallPrintInt("blkIdxBits", blkIdxBits); //TODO remove	
			blockNoPhi->addIncoming(blockNoUpdated, processBlock);
			blocksToDoPhi->addIncoming(blocksToDoUpdated, processBlock);
			numIdxBitsPhi->addIncoming(numIdxBitsUpdated, processBlock);
			idxBitsAccumulatorPhi->addIncoming(idxBitsAccumulatorUpdated, processBlock);

			// If idx pack is mPackWidth wide or if this is last block group, write idx pack to output
			Value * haveFullPack = kb->CreateOr(kb->CreateICmpEQ(numIdxBitsUpdated, kb->getSize(mPackSize))
				, kb->CreateICmpULT(itemsToDo, blockWidth)); 
			kb->CreateCondBr(haveFullPack, writeToOutput, checkLoopCond);
		
			kb->SetInsertPoint(writeToOutput);
			Value * numberPrevWrites = kb->getScalarField("numberOfWrites" + std::to_string(j));
			// What we know: kb->getSize(0) seems to select the first output stream (doesn't work when we make it 1 though...).
			// kb->getInt32 seems to advance within the block by 64 bits. 1 = advance 64 bits, etc.
			kb->CallPrintInt("idxStreams", kb->CreatePtrToInt(idxStreams, kb->getSizeTy()));
			Value * idxStreamPackPtr = kb->CreateBitCast(kb->CreateGEP(idxStreams, {kb->getSize(0), kb->getInt32(j)})
				, iPackPtrTy); // Get a pointer to the jth BitBlock, cast it to a pointer the the first pack within the jth BitBlock
			// field width of idxStream is 1 bit, but we've casted idxSteamBlockPtr to iPackPtrTy (usually 64 bit ptr). 
			// Adding 1 to offset  of idxStreamPackPtr adds 64 bits, not 1 bit!
			Value * idxStreamOutputPtr =  kb->CreateGEP(idxStreamPackPtr, {numberPrevWrites}); // advance in the jth BitBlock to the numberPrevWrites ith pack
			kb->CallPrintInt("idxStreamPackPtr", kb->CreatePtrToInt(idxStreamPackPtr, kb->getSizeTy()));
			kb->CreateBlockAlignedStore(idxBitsAccumulatorUpdated, idxStreamOutputPtr);
						kb->CallPrintRegister("store block", kb->CreateBlockAlignedLoad(kb->CreateGEP(idxStreams, {kb->getSize(0), kb->getInt32(j)})));

			kb->CallPrintInt("idxStreamOutputPtr", kb->CreatePtrToInt(idxStreamOutputPtr, kb->getSizeTy()));
			kb->setScalarField("numberOfWrites" + std::to_string(j), kb->CreateAdd(numberPrevWrites, kb->getSize(1)));
			kb->CallPrintInt("just wrote to output: ", idxBitsAccumulatorUpdated); //TODO remove
			kb->CallPrintInt("that was stream number : ", kb->getSize(j)); //TODO remove
			kb->CallPrintInt("number of previous writes : ", numberPrevWrites); //TODO remove
			idxStreams->getType()->dump();
			Value * resetIdxBits = kb->getSize(0);
			Value * resetNumIdxBits = kb->getSize(0);		
			idxBitsAccumulatorPhi->addIncoming(resetIdxBits, writeToOutput);
			numIdxBitsPhi->addIncoming(resetNumIdxBits, writeToOutput);
			blocksToDoPhi->addIncoming(blocksToDoUpdated, writeToOutput);
			blockNoPhi->addIncoming(blockNoUpdated, writeToOutput);
			kb->CreateBr(checkLoopCond);
				
			kb->SetInsertPoint(terminate);		
			kb->setScalarField("idxBitsAccumulator" + std::to_string(j), idxBitsAccumulatorPhi);
			kb->setScalarField("numIdxBits" + std::to_string(j), numIdxBitsPhi);
			kb->setScalarField("blockNo" + std::to_string(j), blockNoPhi);
			// kb->CallPrintInt("this is the final block: ", kb->CreateICmpULT(itemsToDo, blockWidth)); //TODO remove				
			// kb->CallPrintInt("idxBitsAccumulatorPhi", idxBitsAccumulatorPhi); //TODO remove
			// kb->CallPrintInt("numIdxBitsPhi", numIdxBitsPhi); //TODO remove
	    }
	}

	IdxStreamKernel::IdxStreamKernel(const std::unique_ptr<kernel::KernelBuilder> & kb, unsigned int packSize, unsigned int numInputStreams)
	: MultiBlockKernel("IdxStreamSet"
	, {Binding{kb->getStreamSetTy(numInputStreams, 1), "bitStreams"}}
	// all input streams in IdxStreamSet have same num blocks to process, same fw
	, {Binding{kb->getStreamSetTy(numInputStreams, 1), "idxStreams", FixedRatio(1, packSize)}} 
	, {}
	, {}
	, {/*{Binding{kb->getSizeTy(), "idxBitsAccumulator0"}, Binding{kb->getSizeTy(), "numIdxBits0"}, Binding{kb->getSizeTy(), "numberOfWrites0"}}*/})
	, mPackSize(packSize)
	, mStreamCount(numInputStreams) {
        // TODO assert packsize%blocksize == 0?
        for (unsigned i = 0; i < mStreamCount; i++) {
        	mInternalScalars.emplace_back(kb->getSizeTy(), "idxBitsAccumulator" + std::to_string(i));
        	mInternalScalars.emplace_back(kb->getSizeTy(), "numIdxBits" + std::to_string(i));
        	mInternalScalars.emplace_back(kb->getSizeTy(), "numberOfWrites" + std::to_string(i));
			mInternalScalars.emplace_back(kb->getSizeTy(), "blockNo" + std::to_string(i));
        }
	}
}
