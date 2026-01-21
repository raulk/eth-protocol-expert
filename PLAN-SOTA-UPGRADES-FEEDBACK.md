# Feedback: SOTA Upgrade Plan (Updated)

Review of `PLAN-SOTA-UPGRADES.md` after revisions.

---

## Summary

The plan has been significantly improved. All major concerns from the initial review have been addressed with concrete mitigations. Ready to proceed with Phase 0.

---

## Addressed feedback

| Concern | Resolution | Status |
|---------|------------|--------|
| Label scarcity understated | Added concrete targets (50-100 QA, 20-30 multi-hop), LLM-assisted labeling | ✓ Addressed |
| Versioned eval datasets | Added v1.0 → v1.1 → v2.0 progression | ✓ Addressed |
| Reranking latency | Added top-20 cap, caching, async rerank, single-vendor preference | ✓ Addressed |
| Span extraction complexity | Added edge-case test suite, timeboxed fallback to sentence-level | ✓ Addressed |
| LLM cost for decomposition | Changed to dependency-aware first, LLM for ambiguous only | ✓ Addressed |
| Overlap with existing agents | Added "audit and upgrade existing code before adding new loops" | ✓ Addressed |
| Graph integration missing | Added explicit graph traversal for multi-hop | ✓ Addressed |
| Fallback strategy missing | Added new section with abstain/caveat/speculative modes | ✓ Addressed |
| A/B testing infrastructure | Added shadow mode, feature flags, 10% canary | ✓ Addressed |
| Regression tests | Added regression gates per phase | ✓ Addressed |

---

## Remaining minor suggestions

### 1. Graph modules not explicitly named

Phase 4 mentions "graph traversal explicitly (dependency edges, inferred relations)" but doesn't reference the actual modules:
- `src/graph/eip_graph_builder.py`
- `src/graph/dependency_traverser.py`

Consider adding these to the Implementation Areas list for Phase 4.

### 2. Phase 5 depends on Phase 0 eval logs

Confidence calibration requires sufficient eval data. The plan implies this but doesn't make the dependency explicit. Consider adding: "Requires eval logs from Phases 1-4 runs."

### 3. Trust-tier weights (Phase 6) are undefined

"EIPs > specs > forum" is directionally correct but doesn't specify how weighting works in retrieval scoring. Options:
- Multiplicative factor on similarity scores
- Additive boost to ranking
- Tie-breaker when scores are close

Consider specifying the mechanism or deferring to implementation.

---

## Key improvements in updated plan

### Fallback strategy

Clear default behavior when evidence is insufficient:
- Abstain when below threshold
- Caveat when partial evidence exists
- Speculative mode only on explicit user opt-in

This prevents confident hallucinations.

### Regression gates

Prevents shipping phases that regress quality:
- Phase 1: retrieval recall must not drop
- Phase 2: citation accuracy must improve or stay flat
- Phase 3: latency must stay within budget

### Timeboxed fallbacks

Acknowledges Phase 2 (span extraction) risk with an escape hatch to sentence-level grounding. This is pragmatic—character-level offsets in markdown are genuinely hard.

### Versioned eval datasets

Allows tracking improvements on consistent test sets:
- `v1.0` — initial seed
- `v1.1` — multi-hop expansion
- `v2.0` — span-level gold labels

---

## Overall assessment

The plan is production-ready. The phased approach with measurable targets, regression gates, and explicit fallback strategies demonstrates mature planning.

**Recommendation**: Proceed with Phase 0 (Evaluation Harness).
