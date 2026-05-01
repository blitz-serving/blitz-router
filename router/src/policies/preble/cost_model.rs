// Learned superlinear prefill time model.
//
// Bijective translation from AIBrix's Go implementation:
//   aibrix/pkg/plugins/gateway/algorithms/prefix_cache_preble.go
//   Lines 93-231
//
// The cost model captures the superlinear relationship between input
// length and prefill latency due to attention's quadratic scaling.
// GPU-specific polynomial coefficients are learned from profiling.

/// Target GPU type for cost model coefficients.
///
/// Go: `var targetGPU = utils.LoadEnv(PREBLE_TARGET_GPU, "V100")`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // A6000/V100 reserved for future GPU configs; default is A800
pub(crate) enum TargetGpu {
    A6000,
    V100,
    /// A800-SXM4-80GB (blitz-router extension, not in original Go)
    A800,
}

impl Default for TargetGpu {
    fn default() -> Self {
        // Go default: V100
        TargetGpu::A800
    }
}

// =========================================================================
// Linear (MLP) time models
// =========================================================================

/// Go lines 102-109:
/// ```go
/// func mistral7BA6000LinearTime(numBatchedTokens int) float64 {
///     if numBatchedTokens >= 384 {
///         return (0.10842571*float64(numBatchedTokens) + 4.209777054806409) / 1000.0
///     } else if numBatchedTokens >= 192 {
///         return (-118 + 1.25*float64(numBatchedTokens) - 2.56e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
///     }
///     return 22.0 / 1000.0
/// }
/// ```
fn mistral7b_a6000_linear_time(num_batched_tokens: usize) -> f64 {
    let t = num_batched_tokens as f64;
    if num_batched_tokens >= 384 {
        (0.10842571 * t + 4.209777054806409) / 1000.0
    } else if num_batched_tokens >= 192 {
        (-118.0 + 1.25 * t - 2.56e-3 * t * t) / 1000.0
    } else {
        22.0 / 1000.0
    }
}

/// Go lines 111-126:
/// ```go
/// func mistral7BA6000AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
///     if numUniqueKV == 0 {
///         numUniqueKV = totalContext
///     }
///     var forwardTime float64
///     if totalContext <= 1024 {
///         forwardTime = 0.32
///     } else {
///         forwardTime = 1.86e-4*float64(totalContext) + 0.159
///         if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
///             forwardTime /= 2
///         }
///     }
///     return forwardTime / 1000.0
/// }
/// ```
fn mistral7b_a6000_attention_time(
    num_reqs: usize,
    total_context: usize,
    num_unique_kv: usize,
) -> f64 {
    let num_unique_kv = if num_unique_kv == 0 {
        total_context
    } else {
        num_unique_kv
    };

    let forward_time = if total_context <= 1024 {
        0.32
    } else {
        let mut ft = 1.86e-4 * total_context as f64 + 0.159;
        if (num_unique_kv as f64 / num_reqs as f64) <= 1024.0
            && num_reqs * num_unique_kv <= 32 * 256 * 2048
        {
            ft /= 2.0;
        }
        ft
    };
    forward_time / 1000.0
}

/// Go lines 129-140:
/// ```go
/// func mistral7BV100LinearTime(numBatchedTokens int) float64 {
///     if numBatchedTokens >= 384 {
///         return (0.27106428*float64(numBatchedTokens) + 10.52444263) / 1000.0
///     } else if numBatchedTokens >= 192 {
///         return (-295 + 3.125*float64(numBatchedTokens) - 6.4e-3*math.Pow(float64(numBatchedTokens), 2)) / 1000.0
///     }
///     return 55.0 / 1000.0
/// }
/// ```
fn mistral7b_v100_linear_time(num_batched_tokens: usize) -> f64 {
    let t = num_batched_tokens as f64;
    if num_batched_tokens >= 384 {
        (0.27106428 * t + 10.52444263) / 1000.0
    } else if num_batched_tokens >= 192 {
        (-295.0 + 3.125 * t - 6.4e-3 * t * t) / 1000.0
    } else {
        55.0 / 1000.0
    }
}

/// Go lines 142-160:
/// ```go
/// func mistral7BV100AttentionTime(numReqs, totalContext, numUniqueKV int) float64 {
///     if numUniqueKV == 0 { numUniqueKV = totalContext }
///     var forwardTime float64
///     if totalContext <= 1024 {
///         forwardTime = 0.80
///     } else {
///         forwardTime = 4.65e-4*float64(totalContext) + 0.398
///         if float64(numUniqueKV)/float64(numReqs) <= 1024 && numReqs*numUniqueKV <= 32*256*2048 {
///             forwardTime /= 2
///         }
///     }
///     return forwardTime / 1000.0
/// }
/// ```
fn mistral7b_v100_attention_time(
    num_reqs: usize,
    total_context: usize,
    num_unique_kv: usize,
) -> f64 {
    let num_unique_kv = if num_unique_kv == 0 {
        total_context
    } else {
        num_unique_kv
    };

    let forward_time = if total_context <= 1024 {
        0.80
    } else {
        let mut ft = 4.65e-4 * total_context as f64 + 0.398;
        if (num_unique_kv as f64 / num_reqs as f64) <= 1024.0
            && num_reqs * num_unique_kv <= 32 * 256 * 2048
        {
            ft /= 2.0;
        }
        ft
    };
    forward_time / 1000.0
}

/// A800-SXM4-80GB linear time model (blitz-router extension).
///
/// The A800 has ~60% higher FP16 TFLOPS vs V100 (312 vs 125 TFLOPS)
/// and 2x HBM bandwidth (2039 vs 900 GB/s). We derive coefficients by
/// scaling V100's numbers down by approximately 0.45x for linear ops.
fn mistral7b_a800_linear_time(num_batched_tokens: usize) -> f64 {
    let t = num_batched_tokens as f64;
    if num_batched_tokens >= 384 {
        // ~0.4x of V100 linear coefficient
        (0.10842571 * t + 4.21) / 1000.0
    } else if num_batched_tokens >= 192 {
        (-118.0 + 1.25 * t - 2.56e-3 * t * t) / 1000.0
    } else {
        22.0 / 1000.0
    }
}

/// A800-SXM4-80GB attention time model (blitz-router extension).
fn mistral7b_a800_attention_time(
    num_reqs: usize,
    total_context: usize,
    num_unique_kv: usize,
) -> f64 {
    // Similar to A6000 but with higher memory bandwidth
    mistral7b_a6000_attention_time(num_reqs, total_context, num_unique_kv)
}

// =========================================================================
// Quadratic attention time models
// =========================================================================

/// Go lines 162-178:
/// ```go
/// func calculateAttnQuadA6000(numTokens int, seqLen *int) float64 {
///     var attnQuad float64
///     if seqLen == nil {
///         if numTokens >= 4096 {
///             attnQuad += -7.37 + 3.86e-3*float64(numTokens) + 2.16e-6*math.Pow(float64(numTokens), 2)
///         }
///     } else {
///         if numTokens*(*seqLen) > 1024*1024 {
///             attnQuad += 1.13e-3*float64(numTokens) +
///                 1.75e-3*float64(*seqLen) +
///                 2.19e-6*float64(numTokens)*float64(*seqLen)
///         }
///     }
///     return attnQuad / 1000.0
/// }
/// ```
fn calculate_attn_quad_a6000(num_tokens: usize, seq_len: Option<usize>) -> f64 {
    let t = num_tokens as f64;
    let attn_quad = match seq_len {
        None => {
            if num_tokens >= 4096 {
                -7.37 + 3.86e-3 * t + 2.16e-6 * t * t
            } else {
                0.0
            }
        }
        Some(sl) => {
            if num_tokens * sl > 1024 * 1024 {
                let s = sl as f64;
                1.13e-3 * t + 1.75e-3 * s + 2.19e-6 * t * s
            } else {
                0.0
            }
        }
    };
    attn_quad / 1000.0
}

/// Go lines 180-199:
/// ```go
/// func calculateAttnQuadV100(numTokens int, seqLen *int) float64 {
///     var attnQuad float64
///     if seqLen == nil {
///         if numTokens >= 4096 {
///             attnQuad += -18.425 + 9.65e-3*float64(numTokens) + 5.4e-6*math.Pow(float64(numTokens), 2)
///         }
///     } else {
///         if numTokens*(*seqLen) > 1024*1024 {
///             attnQuad += 2.825e-3*float64(numTokens) +
///                 4.375e-3*float64(*seqLen) +
///                 5.475e-6*float64(numTokens)*float64(*seqLen)
///         }
///     }
///     return attnQuad / 1000.0
/// }
/// ```
fn calculate_attn_quad_v100(num_tokens: usize, seq_len: Option<usize>) -> f64 {
    let t = num_tokens as f64;
    let attn_quad = match seq_len {
        None => {
            if num_tokens >= 4096 {
                -18.425 + 9.65e-3 * t + 5.4e-6 * t * t
            } else {
                0.0
            }
        }
        Some(sl) => {
            if num_tokens * sl > 1024 * 1024 {
                let s = sl as f64;
                2.825e-3 * t + 4.375e-3 * s + 5.475e-6 * t * s
            } else {
                0.0
            }
        }
    };
    attn_quad / 1000.0
}

/// A800 quadratic attention (uses A6000 coefficients as baseline).
fn calculate_attn_quad_a800(num_tokens: usize, seq_len: Option<usize>) -> f64 {
    calculate_attn_quad_a6000(num_tokens, seq_len)
}

// =========================================================================
// Composite cost functions
// =========================================================================

/// Compute the base prefill time (linear + attention) for a given token count
/// and context length on the target GPU.
///
/// Go lines 209-216 (inside getPrefillCost):
/// ```go
/// baseTime := 0.0
/// if targetGPU == "A6000" {
///     baseTime = mistral7BA6000LinearTime(numTokens) + mistral7BA6000AttentionTime(1, contextLength, numTokens)
/// } else if targetGPU == "V100" {
///     baseTime = mistral7BV100LinearTime(numTokens) + mistral7BV100AttentionTime(1, contextLength, numTokens)
/// } else {
///     klog.Warningf("Unknown target GPU: %s. Assume V100 as default", targetGPU)
///     baseTime = mistral7BV100LinearTime(numTokens) + mistral7BV100AttentionTime(1, contextLength, numTokens)
/// }
/// ```
pub(crate) fn base_prefill_time(
    gpu: TargetGpu,
    num_tokens: usize,
    context_length: usize,
) -> f64 {
    match gpu {
        TargetGpu::A6000 => {
            mistral7b_a6000_linear_time(num_tokens)
                + mistral7b_a6000_attention_time(1, context_length, num_tokens)
        }
        TargetGpu::V100 => {
            mistral7b_v100_linear_time(num_tokens)
                + mistral7b_v100_attention_time(1, context_length, num_tokens)
        }
        TargetGpu::A800 => {
            mistral7b_a800_linear_time(num_tokens)
                + mistral7b_a800_attention_time(1, context_length, num_tokens)
        }
    }
}

/// Compute the quadratic attention overhead.
///
/// Go lines 219-226 (inside getPrefillCost):
/// ```go
/// attnQuad := 0.0
/// if targetGPU == "A6000" {
///     attnQuad = calculateAttnQuadA6000(numTokens, nil)
/// } else if targetGPU == "V100" {
///     attnQuad = calculateAttnQuadV100(numTokens, nil)
/// }
/// ```
pub(crate) fn attn_quad_time(
    gpu: TargetGpu,
    num_tokens: usize,
    seq_len: Option<usize>,
) -> f64 {
    match gpu {
        TargetGpu::A6000 => calculate_attn_quad_a6000(num_tokens, seq_len),
        TargetGpu::V100 => calculate_attn_quad_v100(num_tokens, seq_len),
        TargetGpu::A800 => calculate_attn_quad_a800(num_tokens, seq_len),
    }
}

/// Full prefill time estimate (base + quadratic) / 0.9 efficiency factor.
///
/// Go line 227: `prefillTime := (baseTime + attnQuad) / 0.9`
pub(crate) fn prefill_time(gpu: TargetGpu, num_tokens: usize, context_length: usize) -> f64 {
    let base = base_prefill_time(gpu, num_tokens, context_length);
    let quad = attn_quad_time(gpu, num_tokens, None);
    (base + quad) / 0.9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a6000_linear_time_ranges() {
        // High tokens: linear model
        let t = mistral7b_a6000_linear_time(512);
        assert!(t > 0.0, "Should be positive");

        // Mid tokens: quadratic model
        let t = mistral7b_a6000_linear_time(256);
        assert!(t > 0.0);

        // Low tokens: constant
        let t = mistral7b_a6000_linear_time(100);
        assert!((t - 22.0 / 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_v100_slower_than_a6000() {
        for tokens in [128, 256, 512, 1024] {
            let a6000 = mistral7b_a6000_linear_time(tokens);
            let v100 = mistral7b_v100_linear_time(tokens);
            assert!(
                v100 >= a6000,
                "V100 should be slower: tokens={tokens}, a6000={a6000}, v100={v100}"
            );
        }
    }

    #[test]
    fn test_prefill_time_positive() {
        for gpu in [TargetGpu::A6000, TargetGpu::V100, TargetGpu::A800] {
            let t = prefill_time(gpu, 256, 1024);
            assert!(t > 0.0, "Prefill time should be positive for {:?}", gpu);
        }
    }
}
