# TCGA COAD Patient Split Investigation

## TL;DR — Your Hypothesis is **Partially Correct, but for a Different Reason**

Your concern is valid but the root cause is **not that you're training on one "category" and testing on another unrelated one**. All your patients are genuine TCGA COAD patients. The real problem is a well-documented phenomenon called **Tissue Source Site (TSS) batch effects** — and your splits are *not perfectly balanced* across TSS codes.

---

## 1. Understanding TCGA Barcode Naming

Every patient ID follows this structure:

```
TCGA  -  XX  -  YYYY
 ↑        ↑       ↑
Project  TSS   Participant ID
         code   (assigned by BCR)
```

**Example:** `TCGA-AA-3489`
- `TCGA` → The Cancer Genome Atlas project
- `AA` → **Tissue Source Site** = Indivumed (the hospital/institution that collected the sample)
- `3489` → Arbitrary participant number assigned by the Biospecimen Core Resource

The **participant number** (3rd field) is just a sequential ID — whether it's numeric (`3489`) or alphanumeric (`A004`) carries **no clinical or categorical meaning**. Alphanumeric IDs were used for later-enrolled participants at some sites because numeric IDs ran out. It is **NOT** a different patient category.

---

## 2. TSS Code → Institution Mapping (Your 19 Sites)

| TSS Code | Institution | Patients | Train | Val | Test |
|----------|-------------|----------|-------|-----|------|
| **AA** | Indivumed (Germany) | 88 | 54 | 19 | 15 |
| **A6** | Christiana Healthcare (DE, USA) | 43 | 33 | 3 | 7 |
| **CM** | Memorial Sloan Kettering (NY, USA) | 36 | 25 | 6 | 5 |
| **D5** | Asterand (Detroit, USA) | 30 | 20 | 6 | 4 |
| **DM** | University of Michigan | 25 | 15 | 1 | 9 |
| **G4** | Roswell Park Cancer Inst. (NY) | 27 | 20 | 5 | 2 |
| **F4** | Asterand (biorepository) | 16 | 14 | 2 | **0** ⚠️ |
| **CK** | Harvard Medical School | 14 | 13 | 0 | 1 |
| **AZ** | MD Anderson Cancer Center | 19 | 14 | 4 | 1 |
| **AD** | International Genomics Consortium | 13 | 6 | 4 | 3 |
| **NH** | Various | 9 | 6 | 1 | 2 |
| **AY** | Various | 8 | 5 | 2 | 1 |
| **CA** | Vanderbilt / Various | 10 | 8 | 0 | 2 |
| **5M** | Various | 5 | 4 | 0 | 1 |
| **AM** | Cureline | 2 | 2 | **0** | **0** ⚠️ |
| **AU** | Various | 2 | 2 | **0** | **0** ⚠️ |
| **SS** | Various | 1 | 1 | **0** | **0** ⚠️ |
| **T9** | Various | 1 | 1 | **0** | **0** ⚠️ |
| **WS** | Various | 1 | 1 | **0** | **0** ⚠️ |

---

## 3. The Critical Finding: TSS Imbalance Across Splits

### Sites **exclusive to TRAIN** (never seen in val/test):
```
AM, AU, SS, T9, WS
```
These 5 TSS codes (7 patients) appear **only in training** — your val and test sets have never seen images from these institutions.

### Sites **missing from VAL**:
```
5M, AM, AU, CA, CK, SS, T9, WS
```
8 TSS codes are absent from validation — including **CK (Harvard)** which has 14 patients, all in train.

### Sites **missing from TEST**:
```
AM, AU, F4, SS, T9, WS
```
6 TSS codes are absent from test — including **F4 (Asterand)** which has 16 patients, all in train+val.

---

## 4. Why This Explains Your Observations

### ✅ Your hypothesis is **CORRECT in effect** — but the mechanism is:

> **The model learns institution-specific slide preparation artifacts (staining, scanner, tissue processing) during training, then fails on val/test images from different institutional distributions.**

This is called a **TSS batch effect** and is one of the most documented pitfalls in TCGA computational pathology research. Concretely:

| Source of Variation | Effect on Tiles |
|---|---|
| **Different H&E staining protocols** | Color cast differences between institutions |
| **Different tissue fixation** | Morphological texture differences |
| **Different slide scanners** | Resolution, focus, compression artifacts |
| **Different sectioning thickness** | Cell density appearance changes |
| **Different lab technicians** | Stain consistency |

A model trained heavily on AA (Indivumed, 54 train patients) may learn Indivumed's specific staining signature. When it then evaluates on val/test patients from the same site (19 val, 15 test from AA), it appears to "generalize" — but it's actually still overfitting to institutional style.

### The blank tiles problem you found:
Blank/nearly-empty tiles are more common in certain sites due to:
- Different tissue sectioning (some institutions cut thinner sections → more empty slide area)
- Different scanner calibration → overexposed regions appear blank
- Different embedding protocols → air bubbles or edge artifacts

Deleting 53 blank tiles was the **right call**, but this is a symptom of site-specific quality differences.

---

## 5. Numeric vs. Alphanumeric Participant IDs — Verdict

| ID Type | Example | Meaning |
|---|---|---|
| Numeric (`XXXX`) | `TCGA-AA-3489` | Enrolled earlier; number assigned sequentially |
| Alphanumeric (`XXXX`) | `TCGA-AA-A004` | Enrolled later; numeric namespace exhausted |

**There is NO clinical/categorical difference.** The participant ID format is purely an administrative artifact of when/how the BCR ran out of 4-digit numbers for a given TSS. It **does not** indicate a different cancer type, cohort, or data collection era in a meaningful way.

However, your test set has **21 alphanumeric** vs 32 numeric participants, while train has only 53 alphanumeric (22%) — this slight imbalance is a consequence of the TSS distribution, not a separate issue.

---

## 6. What You Should Do

### Immediate Fix: Re-stratify splits by TSS code

```python
# Pseudo-code for stratified splitting by TSS
df['TSS'] = df['PATIENT_ID'].str.split('-').str[1]

# Use stratified split ensuring each TSS appears in all 3 splits
from sklearn.model_selection import train_test_split

# Group by TSS and ensure proportional representation
train, temp = train_test_split(df, test_size=0.3, stratify=df['TSS'], random_state=42)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp['TSS'], random_state=42)
```

> [!WARNING]
> For TSS codes with only 1–2 patients (AM, AU, SS, T9, WS), stratification may not be possible. The best practice is to **pool these rare sites into train only** or apply leave-one-site-out cross-validation.

### Stain Normalization
Apply Macenko or Reinhard stain normalization before tiling:
```python
# Using torchstain or staintools
normalizer.fit(reference_tile)
normalized = normalizer.transform(tile)
```
This reduces color-domain shift between institutions.

### Sanity Check
Run a site-classification experiment: train a simple ResNet to predict TSS code from tile images. If it achieves >70% accuracy, your tiles encode site-specific information that will confound your TMB model.

---

## 7. Summary Verdict

| Question | Answer |
|---|---|
| Are train/val/test from different "categories"? | ❌ No — all are COAD patients from TCGA |
| Is the naming convention difference meaningful? | ❌ No — numeric vs alphanumeric is purely administrative |
| Is there a real distribution mismatch between splits? | ✅ YES — due to TSS (institution) imbalance |
| Does this explain uniform model performance? | ✅ Likely — batch effects are a documented confound |
| Were the blank tiles a problem? | ✅ YES — a symptom of site-specific quality differences |
| What should you fix? | Re-stratify splits by TSS + apply stain normalization |
