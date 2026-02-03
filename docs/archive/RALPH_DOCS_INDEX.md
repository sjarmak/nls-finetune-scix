# Ralph Documentation Index

Complete guide to the extended Ralph workflow for SciX fine-tuning.

## 📋 Quick Navigation

### For First-Time Users
1. Start here: **[QUICK_START.md](QUICK_START.md)** - TL;DR version (5 min read)
2. Then check: **[READINESS_CHECKLIST.md](READINESS_CHECKLIST.md)** - Verify everything is ready (2 min)
3. Run: `./ralph.sh --tool amp 15`

### For Deep Understanding
1. **[WORKFLOW_SUMMARY.md](WORKFLOW_SUMMARY.md)** - Complete overview (10 min)
2. **[YOUR_QUESTIONS_ANSWERED.md](YOUR_QUESTIONS_ANSWERED.md)** - Q&A on all topics (10 min)
3. **[METADATA_FILTERING_EXPLAINED.md](METADATA_FILTERING_EXPLAINED.md)** - How filtering works (15 min)

### For Detailed Specifications
1. **[RALPH_SETUP.md](RALPH_SETUP.md)** - Setup & prerequisites (5 min)
2. **[RALPH_INTEGRATION_GUIDE.md](RALPH_INTEGRATION_GUIDE.md)** - Technical details (20 min)
3. **[prd.json](prd.json)** - User stories with acceptance criteria (reference)
4. **[prompt.md](prompt.md)** - Ralph agent instructions (reference)

---

## 📚 Document Overview

### Foundation Documents (Original Ralph Work - US-001 to US-007)

| Document | Purpose | Status |
|----------|---------|--------|
| `prd.json` | User stories (original 7 + new 8) | ✅ Updated |
| `prompt.md` | Ralph agent instructions | ✅ Updated |
| `ralph.sh` | Automation loop script | ✅ Ready |
| `RALPH_SETUP.md` | Setup & execution guide | ✅ Updated |
| `RALPH_INTEGRATION_GUIDE.md` | Technical integration details | ✅ Updated |

### New Extended Workflow Documents (Ralph Enhancement - US-008 to US-015)

| Document | Purpose | Read Time | Key Audience |
|----------|---------|-----------|--------------|
| **WORKFLOW_SUMMARY.md** | Complete overview: what's done, what's next | 10 min | Everyone |
| **QUICK_START.md** | TL;DR: commands, timelines, 15 test cases | 5 min | Getting started |
| **YOUR_QUESTIONS_ANSWERED.md** | Q&A on workflow expansion, metadata filtering, recommendations | 10 min | Understanding decisions |
| **METADATA_FILTERING_EXPLAINED.md** | How query filtering works, examples, integration | 15 min | Technical deep dive |
| **READINESS_CHECKLIST.md** | Go/no-go checklist before running Ralph | 5 min | Pre-execution |
| **RALPH_DOCS_INDEX.md** | This file - navigation guide | 5 min | Finding what you need |

### Code Files (For Reference)

| File | Purpose | Related Story |
|------|---------|---------------|
| `field_constraints.py` | Field enumeration definitions | US-001 |
| `validate.py` | Constraint validation function | US-002 |
| `constrain.py` | Post-processing filter | US-003 |
| `~/ads-dev/nectar/src/pages/api/nl-search.ts` | API integration | US-006 |
| `scripts/audit_operators.py` | Operator syntax auditor | US-008 |
| `train.py` | Model training (Modal) | US-009 |
| `serve_vllm.py` | Model serving (Modal) | US-010 |

---

## 🎯 What Each Document Answers

### QUICK_START.md
- ❓ How do I run Ralph?
- ❓ What exactly will Ralph do?
- ❓ What are the 15 test cases?
- ❓ How long will this take?

### READINESS_CHECKLIST.md
- ❓ Is everything ready?
- ❓ Do I have all the pieces in place?
- ❓ What could go wrong?
- ❓ Am I ready to start?

### WORKFLOW_SUMMARY.md
- ❓ What was completed in US-001-007?
- ❓ What happens in US-008-015?
- ❓ What's the infrastructure like?
- ❓ What gets delivered?

### YOUR_QUESTIONS_ANSWERED.md
- ❓ Should we expand Ralph to cover the full workflow?
- ❓ Was metadata filtering already implemented?
- ❓ What are your recommendations for expanding the workflow?

### METADATA_FILTERING_EXPLAINED.md
- ❓ How does query filtering work?
- ❓ What metadata constraints are enforced?
- ❓ Show me examples of hallucinations being caught
- ❓ Where is it integrated?

### RALPH_SETUP.md (Original - Updated)
- ❓ How do I install and set up Ralph?
- ❓ What are the prerequisites?
- ❓ How do I run Ralph with Amp or Claude Code?
- ❓ How do I monitor progress?

### RALPH_INTEGRATION_GUIDE.md (Original - Updated)
- ❓ Where will Ralph make changes?
- ❓ How do the pieces integrate?
- ❓ What are the technical details?
- ❓ What's the data flow after Ralph?

---

## 🚀 Recommended Reading Path

### Path A: "I Just Want to Run It" (15 minutes)
1. QUICK_START.md (5 min)
2. READINESS_CHECKLIST.md (5 min)
3. Run: `./ralph.sh --tool amp 15`
4. Follow Ralph's instructions

### Path B: "I Want to Understand What's Happening" (30 minutes)
1. QUICK_START.md (5 min)
2. YOUR_QUESTIONS_ANSWERED.md (10 min)
3. WORKFLOW_SUMMARY.md (10 min)
4. READINESS_CHECKLIST.md (5 min)
5. Run: `./ralph.sh --tool amp 15`

### Path C: "I Need Complete Technical Understanding" (60+ minutes)
1. YOUR_QUESTIONS_ANSWERED.md (10 min)
2. WORKFLOW_SUMMARY.md (10 min)
3. METADATA_FILTERING_EXPLAINED.md (15 min)
4. RALPH_SETUP.md (10 min)
5. RALPH_INTEGRATION_GUIDE.md (20 min)
6. prd.json (reference)
7. READINESS_CHECKLIST.md (5 min)
8. Run: `./ralph.sh --tool amp 15`

---

## 📊 The Complete Workflow at a Glance

```
┌─ PHASE 1: FOUNDATION (US-001 to US-007) ✅ COMPLETE
│  ├─ Field constraints module
│  ├─ Validation functions
│  ├─ Post-processing filter
│  ├─ Training data fixes (536 bare fields)
│  ├─ API integration
│  └─ Documentation
│
├─ PHASE 2: OPERATOR FIX (US-008) 📋 READY
│  ├─ Audit operator syntax
│  ├─ Fix 37 malformed examples
│  └─ Verify no bad patterns remain
│
├─ PHASE 3: RETRAIN (US-009) 📋 READY
│  ├─ Run: scix-finetune train --run-name v3-operators
│  ├─ Training: ~40 min on H100
│  └─ Output: /runs/v3-operators/merged
│
├─ PHASE 4: DEPLOY (US-010) 📋 READY
│  ├─ Run: modal deploy serve_vllm.py
│  ├─ Deployment: ~5 min
│  └─ Endpoint: Modal vLLM live
│
├─ PHASE 5: TEST (US-011 to US-013) 📋 READY
│  ├─ Test 5 operator queries
│  ├─ Test 5 constraint validation edge cases
│  ├─ Test 5 regression tests (original bugs)
│  └─ All 15 must pass
│
├─ PHASE 6: VERIFY (US-014) 📋 READY
│  ├─ Measure latency metrics
│  ├─ Verify syntax validity > 95%
│  ├─ Verify correction rate < 5%
│  └─ Compare before/after
│
└─ PHASE 7: ITERATE (US-015) 📋 READY
   ├─ If tests fail: identify root cause
   ├─ Fix training data or hyperparameters
   ├─ Loop back to US-008 or US-009
   └─ Max 3 loops before escalation

TIMELINE: 3-4 hours total
AUTOMATION: US-008 automatic, US-009-015 guided
STATUS: ✅ Ready to execute
```

---

## 🔑 Key Metrics to Watch

| Metric | Target | Purpose |
|--------|--------|---------|
| Operator accuracy | 100% | All operators generate valid syntax |
| Bare field elimination | 100% | No more unquoted field values |
| Constraint violation rate | 0% | All field values valid |
| Post-processing correction rate | < 5% | Model mostly outputs valid queries |
| Syntax validity | > 95% | Generated queries mostly valid ADS syntax |
| Warm latency | < 0.5s | Model response time |
| Cold start latency | < 1.5s | Inference startup time |
| Error rate (empty results) | < 5% | Query syntax not causing ADS to return 0 results |

---

## 🆘 Troubleshooting Quick Links

### "Ralph seems stuck on a story"
→ See RALPH_SETUP.md - "If Ralph Gets Stuck on a Story"

### "Tests are failing in US-011-014"
→ See RALPH_INTEGRATION_GUIDE.md - "US-015: Iteration Control"

### "I need to understand metadata filtering"
→ See METADATA_FILTERING_EXPLAINED.md

### "I want to see what was already done"
→ See WORKFLOW_SUMMARY.md - "What Was Already Done"

### "Is everything ready?"
→ See READINESS_CHECKLIST.md - "Go/No-Go Decision"

### "What are the 15 test cases?"
→ See QUICK_START.md - "The 3 Test Phases"

---

## 📎 Document Cross-References

### If you're reading...

**QUICK_START.md**
- For timeline details → RALPH_SETUP.md
- For test acceptance criteria → RALPH_INTEGRATION_GUIDE.md
- For infrastructure details → WORKFLOW_SUMMARY.md

**WORKFLOW_SUMMARY.md**
- For how-tos → QUICK_START.md
- For technical specs → RALPH_INTEGRATION_GUIDE.md
- For metadata filtering → METADATA_FILTERING_EXPLAINED.md

**YOUR_QUESTIONS_ANSWERED.md**
- For step-by-step execution → QUICK_START.md
- For complete overview → WORKFLOW_SUMMARY.md
- For setup help → RALPH_SETUP.md

**READINESS_CHECKLIST.md**
- For next steps → QUICK_START.md
- For monitoring → RALPH_SETUP.md
- For failure debugging → RALPH_INTEGRATION_GUIDE.md

---

## ✅ Pre-Execution Checklist

Before running `./ralph.sh --tool amp 15`:

- [ ] You've read QUICK_START.md or READINESS_CHECKLIST.md
- [ ] You've confirmed all items in READINESS_CHECKLIST.md
- [ ] You understand the 15 test cases (from QUICK_START.md)
- [ ] You have 3-4 hours of uninterrupted time
- [ ] You can access `~/ads-dev/nectar` for UI testing (US-011-014)
- [ ] You can run `modal` commands for training/deployment (US-009-010)
- [ ] You have network access to Modal and ADS APIs

---

## 🎬 The Command to Run

When you're ready:

```bash
cd /Users/sjarmak/nls-finetune-scix
./ralph.sh --tool amp 15
```

Ralph will take it from there. Follow the prompts for manual steps (US-009-010, US-011-014).

---

## 📞 Questions?

### About the workflow
→ YOUR_QUESTIONS_ANSWERED.md

### About what's ready
→ READINESS_CHECKLIST.md

### About how to run it
→ QUICK_START.md

### About technical details
→ METADATA_FILTERING_EXPLAINED.md or RALPH_INTEGRATION_GUIDE.md

### About the current status
→ WORKFLOW_SUMMARY.md

---

**Everything is ready. Good luck!** 🚀
