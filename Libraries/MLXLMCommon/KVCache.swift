// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - KVCache Protocol

/// Interface for Key/Value cache for LLMs.
public protocol KVCache {
    /// Get the current offset (total number of tokens processed).
    var offset: Int { get }

    /// Update the cache with new keys and values and return the full cached keys and values.
    /// - Parameters:
    ///   - keys: New keys tensor, typically shape [Batch, Heads, SeqLen, HeadDim]
    ///   - values: New values tensor, typically shape [Batch, Heads, SeqLen, HeadDim]
    /// - Returns: A tuple containing the updated full keys and values tensors up to the current offset.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Check if the cache can be trimmed (typically true for standard/quantized, conditional for rotating).
    func isTrimmable() -> Bool

    /// Trim the cache state by `count` tokens from the end.
    /// Does not shrink allocated memory, only adjusts the logical size (`offset`).
    /// - Parameter count: The number of tokens to trim.
    /// - Returns: The actual number of tokens trimmed (capped by current `offset`).
    func trim(count: Int) -> Int

    // Note: Saving/Loading state requires separate handling, potentially outside the protocol
    // or via specific methods if a unified approach is desired.
    // The Python version uses external functions accessing `state` and `meta_state` properties.
}

// MARK: - Standard Concatenating KVCache

/// A KVCache implementation that concatenates new keys and values along the sequence dimension.
/// Suitable for layers attending to the full sequence history.
public class StandardKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private var currentCapacity: Int = 0  // Track allocated sequence length

    public private(set) var offset = 0
    let step: Int  // Resizing step size

    public init(step: Int = 256) {
        self.step = step
    }

    // Internal state for potential saving (mimics Python's `state` property)
    public var state: (keys: MLXArray?, values: MLXArray?) {
        get {
            if let k = keys, let v = values, offset < currentCapacity {
                // Return only the valid portion
                return (k[0..., 0..., ..<offset, 0...], v[0..., 0..., ..<offset, 0...])
            } else {
                // Return full arrays if offset matches capacity or if nil
                return (keys, values)
            }
        }
        set {
            self.keys = newValue.keys
            self.values = newValue.values
            self.offset = self.keys?.dim(2) ?? 0
            self.currentCapacity = self.offset  // Assume loaded state is exactly the right size
        }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let previousOffset = self.offset
        let newSeqLen = newKeys.dim(2)  // Assuming [B, H, L, D]

        // Check if resizing is needed
        let requiredCapacity = previousOffset + newSeqLen
        let needsResize = keys == nil || requiredCapacity > currentCapacity

        if needsResize {
            let B = newKeys.dim(0)
            let kvHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)

            // Calculate new capacity based on steps
            // Ensure enough space: round up requiredCapacity to the nearest multiple of step
            let nSteps = (requiredCapacity + step - 1) / step
            let newCapacity = nSteps * step

            let kShape = [B, kvHeads, newCapacity, kHeadDim]
            let vShape = [B, kvHeads, newCapacity, vHeadDim]

            // Use `zeros` which might be slightly less efficient than `empty` if available,
            // but ensures initialized memory.
            let resizedK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let resizedV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            // Copy existing data if it exists
            if let currentKeys = self.keys, let currentValues = self.values, previousOffset > 0 {
                // Ensure we only copy the valid part of the old cache
                resizedK[0..., 0..., ..<previousOffset, 0...] =
                    currentKeys[0..., 0..., ..<previousOffset, 0...]
                resizedV[0..., 0..., ..<previousOffset, 0...] =
                    currentValues[0..., 0..., ..<previousOffset, 0...]
            }
            self.keys = resizedK
            self.values = resizedV
            self.currentCapacity = newCapacity
        }

        // Insert the new keys/values into the (potentially resized) cache
        // Use non-optional keys/values now, guarded by the resize logic above.
        self.keys![0..., 0..., previousOffset ..< requiredCapacity, 0...] = newKeys
        self.values![0..., 0..., previousOffset ..< requiredCapacity, 0...] = newValues

        // Update offset
        self.offset = requiredCapacity

        // Return the valid portion of the cache up to the new offset
        return (
            self.keys![0..., 0..., ..<self.offset, 0...],
            self.values![0..., 0..., ..<self.offset, 0...]
        )
    }

    public func isTrimmable() -> Bool {
        true  // Standard cache can always be logically trimmed
    }

    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)
        self.offset -= trimmedCount
        return trimmedCount
    }

    // public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
    //     // Implementation depends on QuantizedKVCache and MLX Swift quantization API
    //     fatalError("Not implemented")
    // }
}

// MARK: - Rotating KVCache

/// A KVCache implementation that uses a fixed-size buffer (`maxSize`) and rotates entries,
/// keeping the first `keep` tokens fixed. Mimics the Python MLX RotatingKVCache.
public class RotatingKVCache: KVCache {

    private var keys: MLXArray?
    private var values: MLXArray?

    public private(set) var offset = 0  // Total tokens processed logically
    private var idx = 0  // Current insertion index within the rotating part of the buffer
    private var currentSize = 0  // Current allocated size (can grow up to maxSize)

    let maxSize: Int
    let keep: Int  // Number of initial tokens to always keep
    let step: Int  // Growth step size when currentSize < maxSize

    /// Initializes a RotatingKVCache.
    /// - Parameters:
    ///   - maxSize: The maximum sequence length to store (the sliding window size).
    ///   - keep: The number of initial tokens to always keep (must be <= maxSize).
    ///   - step: The allocation growth step size.
    public init(maxSize: Int, keep: Int, step: Int = 256) {
        precondition(keep >= 0, "keep must be non-negative")
        precondition(maxSize > 0, "maxSize must be positive")
        precondition(keep <= maxSize, "keep must be less than or equal to maxSize")
        self.maxSize = maxSize
        self.keep = keep
        self.step = step
        self.idx = keep  // Initial insertion point is after the kept tokens
    }

    // Internal state properties for potential saving/loading
    public var state: (keys: MLXArray?, values: MLXArray?) {
        get {
            // Return the cache content in temporal order, sliced to the logical offset
            guard let k = keys, let v = values else { return (nil, nil) }
            let orderedK = temporalOrder(k)
            let orderedV = temporalOrder(v)
            // Slice to the current logical offset
            return (orderedK[0..., 0..., ..<offset, 0...], orderedV[0..., 0..., ..<offset, 0...])
        }
        set {
            // Loading state is complex because the internal representation might not be
            // temporally ordered. We'd need meta_state (idx, currentSize) to restore properly.
            // Simple restoration assuming the input is temporally ordered and fits maxSize:
            self.keys = newValue.keys
            self.values = newValue.values
            self.offset = self.keys?.dim(2) ?? 0
            self.currentSize = self.offset  // This might not be correct if offset > maxSize
            self.idx = self.offset  // This is likely incorrect for a loaded rotating cache
            // Proper loading requires meta_state.
            print("Warning: Simple state restoration for RotatingKVCache might be inaccurate.")
        }
    }

    // Metadata needed for proper saving/loading (mimics Python meta_state)
    public var metaState:
        (keep: Int, maxSize: Int, step: Int, offset: Int, idx: Int, currentSize: Int)
    {
        (keep, maxSize, step, offset, idx, currentSize)
    }

    // Helper to rearrange the buffer into temporal order
    // Note: This creates copies and can be expensive. Only use for saving or multi-token updates.
    private func temporalOrder(_ v: MLXArray) -> MLXArray {
        let bufferSeqLen = v.dim(2)
        if idx == bufferSeqLen || bufferSeqLen <= keep {
            // Already in order or only contains kept tokens
            return v[0..., 0..., ..<min(offset, bufferSeqLen), 0...]
        } else if idx < offset {  // Check if rotation has occurred and idx is valid index
            // Rotated: [keep | rotated_part_2 | rotated_part_1 ]
            // Order:   [keep | rotated_part_1 | rotated_part_2 ]
            let part1 = v[0..., 0..., keep ..< idx, 0...]
            let part2 = v[0..., 0..., idx ..< bufferSeqLen, 0...]
            var components = [MLXArray]()
            if keep > 0 {
                components.append(v[0..., 0..., ..<keep, 0...])
            }
            components.append(part1)
            if part2.dim(2) > 0 {  // Only append if part2 exists
                components.append(part2)
            }
            return MLX.concatenated(components, axis: 2)[
                0..., 0..., ..<min(offset, bufferSeqLen), 0...]
        } else {
            // Not fully rotated yet, or offset is within idx
            return v[0..., 0..., ..<min(offset, bufferSeqLen), 0...]
        }
    }

    // Helper to trim the middle part (between keep and start of insertion)
    // This modifies the array in place conceptually for the update logic.
    // Returns the new array after trimming.
    private func trimMiddle(trimSize: Int, v: MLXArray) -> MLXArray {
        if trimSize <= 0 { return v }
        // Keep: v[..., :keep, :]
        // Keep: v[..., keep+trimSize:, :]
        var components = [MLXArray]()
        if keep > 0 {
            components.append(v[0..., 0..., ..<keep, 0...])
        }
        let remainingStart = keep + trimSize
        if remainingStart < v.dim(2) {
            components.append(v[0..., 0..., remainingStart..., 0...])
        }
        if components.isEmpty {
            // Should not happen if v was not empty, but handle defensively
            return v  // Or return an empty array of correct shape/dtype?
        } else {
            return MLX.concatenated(components, axis: 2)
        }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let S = newKeys.dim(2)  // Sequence length of the update

        if S == 1 {
            return updateInPlace(keys: newKeys, values: newValues)
        } else {
            return updateConcat(keys: newKeys, values: newValues)
        }
    }

    // Handles single-token updates (efficient in-place rotation)
    private func updateInPlace(keys newKey: MLXArray, values newValue: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let S = 1  // Single token update

        // 1. Grow buffer if needed and not yet at max size
        let needsGrow = keys == nil || (offset >= currentSize && currentSize < maxSize)
        if needsGrow {
            let B = newKey.dim(0)
            let kvHeads = newKey.dim(1)
            let kHeadDim = newKey.dim(3)
            let vHeadDim = newValue.dim(3)

            // Determine growth size
            let growth = min(step, maxSize - currentSize)
            let newBufferSeqLen = currentSize + growth

            let kShape = [B, kvHeads, newBufferSeqLen, kHeadDim]
            let vShape = [B, kvHeads, newBufferSeqLen, vHeadDim]

            let grownK = MLXArray.zeros(kShape, dtype: newKey.dtype)
            let grownV = MLXArray.zeros(vShape, dtype: newValue.dtype)

            if let currentKeys = self.keys, let currentValues = self.values, currentSize > 0 {
                grownK[0..., 0..., ..<currentSize, 0...] = currentKeys
                grownV[0..., 0..., ..<currentSize, 0...] = currentValues
            }
            self.keys = grownK
            self.values = grownV
            self.currentSize = newBufferSeqLen
            // `idx` should already point to the next insertion spot (`offset` if no rotation happened yet)
            // If we just grew, idx should be `currentSize` before growth.
            self.idx = offset  // Reset idx to current offset after growth before rotation logic
        }

        // Ensure keys/values exist now
        guard var currentKeys = self.keys, var currentValues = self.values else {
            fatalError("Cache allocation/growth failed")
        }

        // 2. Check if rotation within the buffer is needed
        // This happens when the buffer is full (currentSize == maxSize)
        // and the insertion index `idx` reaches the end.
        if idx == currentSize && currentSize == maxSize {
            idx = keep  // Wrap around insertion point to after the kept tokens
        }

        // 3. Insert the new token
        currentKeys[0..., 0..., idx ..< (idx + S), 0...] = newKey
        currentValues[0..., 0..., idx ..< (idx + S), 0...] = newValue

        // 4. Update pointers
        offset += S
        idx += S

        self.keys = currentKeys  // Update struct's stored arrays
        self.values = currentValues

        // 5. Return the relevant cache view
        // If the buffer isn't full according to the logical offset, return the slice
        if offset < currentSize {
            return (
                currentKeys[0..., 0..., ..<offset, 0...], currentValues[0..., 0..., ..<offset, 0...]
            )
        } else {
            // Return the full buffer (or the relevant slice if offset > currentSize somehow?)
            // Python returns the full buffer here.
            return (currentKeys, currentValues)
        }
    }

    // Handles multi-token updates (less efficient, involves reordering/concatenation)
    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        guard var currentKeys = self.keys, var currentValues = self.values else {
            // First update
            self.keys = newKeys
            self.values = newValues
            self.offset = newKeys.dim(2)
            self.currentSize = self.offset
            self.idx = self.offset
            return (self.keys!, self.values!)
        }

        let S = newKeys.dim(2)

        // 1. Put current cache in temporal order
        // This is expensive but necessary to mimic Python's logic for concat
        currentKeys = temporalOrder(currentKeys)
        currentValues = temporalOrder(currentValues)

        // 2. Calculate trim size
        // How many tokens to remove from the middle (after `keep`) to make space
        // such that the final size doesn't exceed maxSize.
        let potentialNewOffset = offset + S
        let trimSize = max(0, (currentKeys.dim(2) + S) - maxSize)  // Trim needed if current+new > max
        // Python logic seems slightly different: `trim_size = self._idx - self.max_size`
        // Let's stick to the goal: final size <= maxSize
        let effectiveTrimSize = max(0, potentialNewOffset - maxSize)  // How many elements exceed maxSize

        // 3. Trim the middle part
        // Remove `effectiveTrimSize` elements starting after the `keep` index
        if effectiveTrimSize > 0 {
            // We need a function similar to Python's _trim that removes from the middle
            // Let's define a simplified trim: remove oldest rotating elements
            let startIndexToKeep = keep + effectiveTrimSize
            if startIndexToKeep < currentKeys.dim(2) {
                var keepComponentsK = [MLXArray]()
                var keepComponentsV = [MLXArray]()
                if keep > 0 {
                    keepComponentsK.append(currentKeys[0..., 0..., ..<keep, 0...])
                    keepComponentsV.append(currentValues[0..., 0..., ..<keep, 0...])
                }
                keepComponentsK.append(currentKeys[0..., 0..., startIndexToKeep..., 0...])
                keepComponentsV.append(currentValues[0..., 0..., startIndexToKeep..., 0...])

                currentKeys = MLX.concatenated(keepComponentsK, axis: 2)
                currentValues = MLX.concatenated(keepComponentsV, axis: 2)
            } else if keep > 0 {
                // Trimmed everything except the 'keep' part
                currentKeys = currentKeys[0..., 0..., ..<keep, 0...]
                currentValues = currentValues[0..., 0..., ..<keep, 0...]
            } else {
                // Trimmed everything
                // Need to handle shape correctly - create empty arrays?
                let B = newKeys.dim(0)
                let H = newKeys.dim(1)
                let Dk = newKeys.dim(3)
                let Dv = newValues.dim(3)
                currentKeys = MLXArray.zeros([B, H, 0, Dk], dtype: newKeys.dtype)
                currentValues = MLXArray.zeros([B, H, 0, Dv], dtype: newValues.dtype)
            }
        }

        // 4. Concatenate new keys/values
        self.keys = MLX.concatenated([currentKeys, newKeys], axis: 2)
        self.values = MLX.concatenated([currentValues, newValues], axis: 2)

        // 5. Update pointers
        self.offset += S
        self.currentSize = self.keys!.dim(2)  // Update current size after concat
        self.idx = self.currentSize  // After concat, idx points to the end

        return (self.keys!, self.values!)
    }

    public func isTrimmable() -> Bool {
        // Python logic: return self.offset < self.max_size
        // This seems counter-intuitive. Trimming usually means removing past tokens.
        // Let's assume trimming means reducing the logical offset.
        true  // Can always reduce logical offset
    }

    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)
        self.offset -= trimmedCount
        // We also need to adjust `idx` if it's within the trimmed region,
        // but Python's trim only adjusts offset. Let's match that for now.
        // self.idx = max(self.keep, self.idx - trimmedCount) // A possible adjustment
        return trimmedCount
    }

    // public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
    //     fatalError("RotatingKVCache Quantization NYI")
    // }
}

// MARK: - Helper Functions

/// Creates an additive causal mask for attention.
///
/// Creates mask for `[B, H, N, N + Offset]` where `N` is `n`.
///
/// - Parameters:
///   - n: The sequence length of the query.
///   - offset: The offset for the key/value sequence length.
/// - Returns: An MLXArray suitable for adding to attention scores.
public func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
    let queryIndices = MLXArray(Int32(offset) ..< Int32(offset + n))  // Shape [N]
    let keyIndices = MLXArray(Int32(0) ..< Int32(offset + n))  // Shape [N + Offset]

    // Compare queryIndices[i] < keyIndices[j]
    // queryIndices shape: [N, 1]
    // keyIndices shape:   [1, N + Offset]
    // Result shape:       [N, N + Offset]
    let mask = queryIndices.expandedDimensions(axis: 1) .< keyIndices.expandedDimensions(axis: 0)

    // Add dimensions for batch and head: [1, 1, N, N + Offset]
    let mask4D = mask.expandedDimensions(axes: [0, 1])

    // Use a large negative number for masked positions
    // Multiplying by float directly handles type promotion if mask is bool
    return mask4D * Float(-1e9)
}

/// Creates an attention mask based on input sequence length and cache offset.
///
/// Only creates a mask for multi-token inputs (prefill phase). For single-token
/// generation (t=1), no explicit mask is typically needed as attention is only
/// computed for the single query token against all keys.
///
/// - Parameters:
///   - h: The input tensor to the attention layer. Expected shape [Batch, SeqLen, HiddenDim]
///        or [Batch, Heads, SeqLen, HeadDim]. `SeqLen` is extracted from `dim(1)` or `dim(2)`.
///   - cache: An array of KVCache instances (one per layer). Used to determine the offset.
///            Assumes all caches have the same offset.
///   - seqLenDim: The dimension index representing sequence length in `h`. Typically 1 or 2.
/// - Returns: An optional MLXArray attention mask, or nil if `t <= 1`.
public func createAttentionMask(h: MLXArray, cache: [any KVCache]?, seqLenDim: Int = 1) -> MLXArray?
{
    guard h.ndim > seqLenDim else {
        // Handle cases where input tensor doesn't have expected dimensions
        print(
            "Warning: Input tensor `h` has fewer dimensions than expected (\(h.ndim) vs \(seqLenDim + 1)). Cannot determine sequence length."
        )
        return nil
    }
    let t = h.dim(seqLenDim)  // Extract sequence length

    // Only create mask for multi-token inputs (prefill)
    if t > 1 {
        var offset = 0
        // Get offset from the first cache entry
        if let firstCache = cache?.first {
            offset = firstCache.offset
        }
        // Use the refined createAdditiveCausalMask which returns 4D mask
        return createAdditiveCausalMask(n: t, offset: offset).asType(h.dtype)
    }
    // No mask needed for single-token generation
    return nil
}

// MARK: - KVCacheSimple

public class KVCacheSimple: KVCache, Evaluatable {
    var keys: MLXArray?
    var values: MLXArray?

    public var offset = 0
    var step = 256  // Resizing step size

    public init() {}

    public func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let newSeqLen = keys.dim(2)  // Assuming [B, H, L, D]

        // Check if resizing is needed
        let needsResize: Bool
        if let currentKeys = self.keys {
            needsResize = (previous + newSeqLen) > currentKeys.dim(2)  // Check if new length exceeds current capacity
        } else {
            needsResize = true  // Needs allocation if keys is nil
        }

        if needsResize {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            // Calculate new size based on steps
            let requiredLength = previous + newSeqLen
            let nSteps = (requiredLength + step - 1) / step  // Number of steps needed
            let newCapacity = nSteps * step

            let kShape = [B, kvHeads, newCapacity, kHeadDim]
            let vShape = [B, kvHeads, newCapacity, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            // Copy existing data if it exists
            if let currentKeys = self.keys, let currentValues = self.values, previous > 0 {
                // Copy only the valid part of the old cache
                newK[0..., 0..., ..<previous, 0...] = currentKeys[0..., 0..., ..<previous, 0...]
                newV[0..., 0..., ..<previous, 0...] = currentValues[0..., 0..., ..<previous, 0...]
            }
            self.keys = newK
            self.values = newV
        }

        // Insert the new keys/values
        // Use optional chaining just in case allocation failed, though it shouldn't
        self.keys?[0..., 0..., previous ..< (previous + newSeqLen), 0...] = keys
        self.values?[0..., 0..., previous ..< (previous + newSeqLen), 0...] = values

        // Update offset
        self.offset += newSeqLen

        // Return the valid portion of the cache up to the new offset
        // Guard against nil arrays before slicing and returning
        guard let currentKeys = self.keys, let currentValues = self.values else {
            // This should ideally not happen if allocation succeeded or was not needed
            fatalError("Cache arrays are unexpectedly nil after update.")
        }

        return (
            currentKeys[0..., 0..., ..<self.offset, 0...],
            currentValues[0..., 0..., ..<self.offset, 0...]
        )
    }

    /// Checks if the cache can be logically trimmed. Always true for simple cache.
    public func isTrimmable() -> Bool {
        return true
    }

    /// Trims the cache state logically by reducing the offset.
    /// Does not shrink allocated memory.
    /// - Parameter count: The number of tokens to trim from the end.
    /// - Returns: The actual number of tokens trimmed.
    public func trim(count: Int) -> Int {
        let trimmedCount = min(self.offset, count)  // Don't trim more than available
        self.offset -= trimmedCount
        return trimmedCount
    }
}
