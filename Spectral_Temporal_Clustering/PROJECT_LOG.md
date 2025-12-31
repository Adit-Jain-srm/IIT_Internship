# STC Hand Gesture Recognition - Project Development Log

**Project**: Spectral Temporal Clustering for Hand Gesture Recognition

---

## Executive Summary

This document tracks development, issues, resolutions, and current state of the STC implementation. Progress: **36.5625% → 45.625%** accuracy (+9.0625 percentage points) through systematic optimization.

**Current Status**: ✅ **DTW Implementation Complete** ❌ **Testing Shows Regression** - **RECOMMEND DISABLING**

---

## Project Timeline & Milestones

| Milestone | Status | Accuracy |
|-----------|--------|----------|
| Initial Implementation | ✅ Complete | 36.5625% |
| A/B Testing - Temporal Features | ✅ Complete | 43.75% |
| Parameter Optimization | ✅ Complete | 44.0625% |
| Temporal Graph Sparsification | ✅ Complete | **45.625%** ✅ **BEST** |
| DTW Implementation | ✅ Complete | - |
| DTW Testing & Evaluation | ✅ Complete | **37.5%** ❌ **REGRESSION** |

---

## Issues Encountered & Resolutions

### Issue #1: Incorrect Video Segmentation ✅ **RESOLVED**

**Severity**: Critical  
**Impact**: Only extracting 7-8 videos instead of 320

**Problem**: Sliding window approach instead of fixed-length chunks

**Resolution**: Rewrote segmentation using fixed 150-frame chunks with proper frame counting

**Result**: ✅ Successfully extracts all 320 video sequences

**Code**: `segment_sequences_fixed()` function with validation

---

### Issue #2: Temporal Graph Too Dense (99.69%) ✅ **RESOLVED**

**Severity**: Critical  
**Impact**: Temporal component ineffective

**Problem**: 99.69% density (nearly fully connected), `n_neighbors_temporal` ineffective

**Resolution**: Implemented similarity threshold-based sparsification (threshold=0.3)

**Optimization**: Tested 32 configurations, best: `similarity_threshold=0.3`
- Graph density: 99.69% → 18.18%
- Accuracy: 44.0625% → 45.625% (+1.5625%)

**Result**: ✅ Temporal graph sparse and meaningful

**Code**: Modified `_build_temporal_graph()` with sparsification logic

---

### Issue #3: Gesture-Specific Failures 🔄 **IN PROGRESS**

**Severity**: High  
**Impact**: Three gestures with zero/low accuracy

**Failures**:
- Good: 0% (all → Cleaning)
- Pick: 0% (distributed)
- Wave: 12.5% (mostly → Cleaning, Come)
- Come: 35% (50% → Cleaning)

**Root Causes**:
1. Similar static poses (Good vs Cleaning - both double-hand)
2. Missing sequence alignment (timing variations)
3. Insufficient temporal differentiation

**Resolution Attempted**: ✅ DTW implemented, pending testing

**Pending**: Good vs Cleaning (needs specialized features), Wave (needs FFT)

---

### Issue #4: Poor Temporal Feature Weights ✅ **RESOLVED**

**Severity**: Medium  
**Impact**: Temporal features over-weighted (15% static, 85% temporal)

**Resolution**: A/B testing found optimal: **Balanced weights (50% static, 50% temporal)**
- Accuracy: 38.75% → 43.75% (+5%)

**Result**: ✅ Balanced weights confirmed optimal

---

### Issue #5: Naive Prediction Method ✅ **RESOLVED**

**Severity**: Medium  
**Impact**: Evaluation not using learned embedding

**Resolution**: Implemented Nyström extension for spectral projection

**Result**: ✅ Prediction uses learned spectral embedding

**Code**: Modified `predict()` method with spectral projection

---

### Issue #6: Suboptimal Alpha ✅ **RESOLVED**

**Severity**: Medium  
**Impact**: Default alpha=0.5 not optimal

**Resolution**: Grid search found optimal: **alpha=0.3 (70% temporal, 30% spatial)**
- Accuracy: 43.75% → 44.0625%

**Result**: ✅ Optimal alpha confirmed

---

### Issue #7: Suboptimal Spatial Neighbors ✅ **RESOLVED**

**Severity**: Low-Medium  
**Impact**: Default spatial neighbors=10 not optimal

**Resolution**: Testing found optimal: **spatial neighbors = 5**

**Result**: ✅ Optimal spatial neighbors confirmed

---

### Issue #8: Missing Sequence Alignment ✅ **IMPLEMENTED** ❌ **REGRESSION OBSERVED**

**Severity**: High  
**Impact**: Gestures with timing variations penalized incorrectly

**Resolution**: ✅ Implemented DTW (Dynamic Time Warping)
- `_dtw_distance()`: Computes DTW distance
- `_dtw_align_sequences()`: Aligns sequences before feature extraction
- Added `use_dtw` and `dtw_radius` parameters

**Test Results**: ❌ **REGRESSION**
- **Overall Accuracy**: 45.625% → 37.5% (-8.125 percentage points)
- **Emergency_calling**: 97.5% → 0% (all misclassified as Cleaning) ❌ **CRITICAL**
- **Come**: 35% → 50% (+15%) ✅ **Improved**
- **Wave**: 12.5% → 27.5% (+15%) ✅ **Improved**
- **Pick**: 0% → 2.5% (+2.5%) ⚠️ **Minimal improvement**

**Root Cause** (See `DTW_EMERGENCY_CALLING_INVESTIGATION.md` for detailed analysis):
- ✅ **CONFIRMED**: DTW makes Emergency_calling sequences more similar to each other (+0.0477)
- ✅ **CONFIRMED**: This creates a tighter Emergency_calling cluster (+25 intra-cluster connections: 137 → 162)
- ✅ **CONFIRMED**: The tighter cluster changes the graph Laplacian, affecting spectral embedding
- ✅ **CONFIRMED**: In spectral space, the tighter Emergency_calling cluster merges with Cleaning cluster
- **Key Insight**: The issue is NOT direct pairwise similarity, but **graph structure changes** that affect spectral clustering. Emergency_calling vs Cleaning similarity only increases by +0.0050, but the graph structure changes cause cluster merging.

**Status**: ✅ Implemented ❌ **CAUSES REGRESSION** - ✅ **DISABLED** (use_dtw=False in notebook)

**Recommendation**: 
- **DISABLE DTW** (use_dtw=False) - Use optimal configuration without DTW
- **Alternative**: Implement gesture-specific DTW (only for Come, Wave, Pick)
- **Investigation Needed**: Why does DTW destroy Emergency_calling recognition?

---

## Current State

### ✅ Completed Features

1. **Data Loading**: Fixed segmentation (320 sequences), zero-padding removal, normalization
2. **STC Algorithm**: Spatial/temporal graphs, Laplacians, spectral decomposition, K-Means
3. **Optimization**: Balanced weights, alpha=0.3, spatial_k=5, threshold=0.3
4. **Prediction**: Spectral projection (Nyström extension), Hungarian mapping
5. **DTW**: Distance computation, sequence alignment, integration
6. **Analysis**: Comprehensive documentation, gesture-wise analysis, metrics

### ⏳ Pending

1. **DTW Testing**: Evaluate impact, optimize parameters
2. **Gesture-Specific**: Good vs Cleaning, Wave improvements
3. **Advanced Features**: FFT, hand proximity, motion direction

---

## Current Performance Metrics

### WITHOUT DTW (Optimal Configuration) ✅ **BEST**

**Overall Accuracy**:
- STC: **45.625%** (146/320)
- GMM: 43.75% (140/320)
- Improvement: +1.875 percentage points

**Clustering Quality**:
- Silhouette: 0.6634 (vs GMM: 0.5335) - **+24.35%**
- Davies-Bouldin: 0.4263 (vs GMM: 0.9114) - **+53.22%**
- Calinski-Harabasz: 3666.38 (vs GMM: 3453.01) - **+6.18%**

**Per-Gesture Accuracy (STC)**:
- ✅ Cleaning: 100.00%
- ✅ Emergency_calling: 97.50%
- ✅ Stack: 70.00%
- ⚠️ Give: 50.00%
- ⚠️ Come: 35.00%
- ❌ Wave: 12.50%
- ❌ Good: 0.00%
- ❌ Pick: 0.00%

### WITH DTW (Current - REGRESSION) ❌

**Overall Accuracy**:
- STC: **37.5%** (120/320) ❌ **-8.125% from optimal**
- GMM: 43.75% (140/320)
- **STC now UNDERPERFORMS GMM by 6.25 percentage points**

**Clustering Quality** (with DTW):
- Silhouette: 0.7000 (vs GMM: 0.5335) - **+31.22%** (improved)
- Davies-Bouldin: 0.4915 (vs GMM: 0.9114) - **+46.08%** (worse than without DTW)
- Calinski-Harabasz: 3482.46 (vs GMM: 3453.01) - **+0.85%** (worse than without DTW)

**Per-Gesture Accuracy (STC with DTW)**:
- ✅ Cleaning: 100.00% (no change)
- ❌ Emergency_calling: 0.00% ❌ **CRITICAL REGRESSION** (was 97.5%)
- ✅ Stack: 70.00% (no change)
- ⚠️ Give: 50.00% (no change)
- ✅ Come: 50.00% ✅ **+15% improvement**
- ❌ Wave: 27.50% ✅ **+15% improvement** (but still poor)
- ❌ Good: 0.00% (no change)
- ❌ Pick: 2.50% ⚠️ **+2.5% minimal improvement**

**Optimal Configuration** (RECOMMENDED):
- Alpha: 0.3 (70% temporal, 30% spatial)
- Spatial neighbors: 5
- Temporal threshold: 0.3
- Temporal weights: Balanced (50/50)
- **DTW: DISABLED** ❌ (causes regression)

---

## Technical Architecture

**Spatial Graph**: k-NN (k=5), 2.06% density ✅  
**Temporal Graph**: Threshold-based (0.3), 18.18% density, DTW enabled ✅  
**Joint Laplacian**: α=0.3 (30% spatial, 70% temporal) ✅  
**Spectral Decomposition**: 8 smallest eigenvalues/eigenvectors ✅  
**Prediction**: Nyström extension ✅

---

## Known Limitations

1. **Computational Complexity**: DTW O(n²) per pair - use Sakoe-Chiba band if slow
2. **Gesture Failures**: Good (0%), Pick (0%), Wave (12.5%) - DTW should help Pick/Wave
3. **Accuracy Gap**: Current 45.625% vs Target 70-80% (~25-35 point gap)
4. **Limited Features**: No FFT, hand proximity, or gesture-specific features

---

## Performance Improvement Journey

```
36.5625%  (Initial - wrong weights)
    ↓ +7.1875%
43.75%    (Balanced weights)
    ↓ +0.3125%
44.0625%  (Parameter optimization)
    ↓ +1.5625%
45.625%   (Temporal graph sparsification) ✅ BEST
    ↓ -8.125%
37.5%     (DTW - REGRESSION) ❌
```

**Total Improvement**: +9.0625 percentage points (without DTW)  
**Current with DTW**: +0.9375 percentage points (from initial)  
**Recommendation**: **DISABLE DTW** - Use 45.625% configuration  
**Target**: 70-80%  
**Remaining Gap**: ~24-35 percentage points (from optimal 45.625%)

---

## Improvements

### Immediate
1. ✅ **DTW tested** - Results show regression (45.625% → 37.5%)
2. ✅ **DTW DISABLED** - Updated notebook to use_dtw=False
3. ✅ **Emergency_calling failure investigated** - Root cause identified (graph structure changes)
4. ✅ **Priority 3 testing added** - Cell 9 tests lower alpha values (0.2, 0.1, 0.0)
5. Address Good gesture (needs specialized features, DTW doesn't help)

### Short-term (1-2 weeks)
1. ✅ **DTW testing complete** - Regression observed, recommend disabling
2. **Implement gesture-specific DTW** - Use DTW only for Come, Wave, Pick (not Emergency_calling)
3. Test lower alpha values (0.2, 0.1, 0.0) with DTW disabled
4. Fine-tune threshold around 0.3
5. Investigate why DTW causes Emergency_calling to fail completely

### Medium-term (2-4 weeks)
1. Implement specialized features (hand proximity, FFT)
2. Address Good vs Cleaning confusion
3. Improve Wave gesture recognition

### Long-term (1-2 months)
1. Gesture-specific models (single-hand vs double-hand)
2. Advanced methods (Graph Neural Networks, deep learning)
3. Semi-supervised learning

---

## Lessons Learned

1. **Graph sparsification critical**: Dense graphs (99.69%) provide minimal information
2. **Balance is key**: 50/50 static-temporal outperforms extremes
3. **Temporal > Spatial**: Lower alpha (more temporal) performs better
4. **Systematic optimization works**: Grid search found optimal parameters
5. **Gesture-specific issues exist**: Some gestures need specialized attention
6. **DTW important**: Temporal alignment handles speed/timing variations
7. **Always validate**: Check data loading, graph properties, per-gesture accuracy
8. **Documentation essential**: Helps track progress and plan next steps

---

**Document Version**: 2.0  
**Last Updated**: After DTW testing - **REGRESSION OBSERVED**  
**Key Finding**: DTW causes 8.125% accuracy regression (45.625% → 37.5%)  
**Recommendation**: **DISABLE DTW** - Use optimal configuration (45.625% accuracy)  
**Next Review**: After investigating DTW failure or implementing gesture-specific DTW
