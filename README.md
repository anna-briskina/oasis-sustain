# Does Alzheimer's spread the same way in everyone?

This project uses brain scans and machine learning to ask two questions about Alzheimer's
disease:

1. **Does the damage always spread through the brain in the same order, or are there
   different paths?**
2. **From a single scan, can we tell how far along that path a person is?**

*This is the plain-language overview. For methods, statistics and how to run the code, see
the [technical write-up](TECHNICAL.md).*

> This is a research and learning project, not a medical tool. It cannot diagnose anyone.

---

## The idea

In Alzheimer's disease, parts of the brain slowly shrink as brain cells are lost. You can
picture it like rust spreading across a car: it starts somewhere, then gradually reaches
more and more parts.

The difficulty is that we usually only have **one scan per person**, a single snapshot,
not a film. But if we look at many people who are at different points in the disease,
we can piece together the order in which the damage tends to spread. It is a bit like
working out how a story unfolds from a pile of photos taken at different moments.

We used a method called **SuStaIn** (Subtype and Stage Inference) that does exactly this.
It looks at the snapshots and works out:

- **the path** (or paths) the damage takes through the brain, and
- **how far along** their path each person is.

## The data

We used **OASIS-1**, a free, public collection of brain MRI scans. We studied 190 adults
aged 60 and over: about half healthy, and half with very mild to moderate dementia. For each
person we measured how much brain tissue was left in 13 regions known to be affected by
Alzheimer's.

Importantly, **the model never saw anyone's diagnosis or test scores.** It worked out
everything from the scans alone. We only used the clinical information afterwards, to
check whether its answers made sense.

## What we found

### 1. There are three different paths

The damage does not spread the same way in everyone. The model found three paths:

| Path | Where the damage starts | Share of patients with measurable damage |
|---|---|---|
| **Temporal-first** | the sides of the brain (temporal lobes), then the memory centre | about 1 in 3 |
| **Parietal-first** | the top and back of the brain; the memory centre is reached later | about 1 in 3 |
| **Hippocampus-first** | right in the memory centre (the hippocampus), the "classic" pattern | about 1 in 3 |

![The three paths. Each panel is one path; rows are brain regions, and the dark squares show
when each region is affected, from earliest (left) to latest (right).](docs/figures/pvd_3subtypes.png)

*How to read this picture: each of the three panels is one path. Each row is a brain region,
and moving left to right is moving forward in the disease. A dark square means "this region
is affected at this point". In each panel, the regions at the top are hit first.*

Researchers studying much larger groups have described very similar patterns before, which
suggests the model found something real.

### 2. The "how far along" score works

The model gives everyone a **stage**, a bit like a progress bar along their path. Stage 0
means no measurable damage; the higher the stage, the more of the path has been covered.

We then compared the stages with each person's dementia rating from their doctor, which the
model had never seen:

| Doctor's dementia rating | None | Very mild | Mild | Moderate |
|---|---|---|---|---|
| Typical stage from the model | 0 | 4 | 7 | 13 |

![Left: stage for each level of the doctor's rating. Right: stage against a memory test
score.](docs/figures/stage_vs_clinical.png)

*How to read this picture: on the left, each box is a group of people with the same doctor's
rating, and the line in the middle of the box is the typical stage. The boxes climb from left
to right: the sicker the person, the further along the progress bar. On the right, each dot
is a person; people further to the right (higher stage) tend to have lower memory-test scores.*

The progress bar is about disease, not age: among healthy people, older people did not get
higher stages.

### 3. The results hold up

A result that changes when you change one small thing is not worth much, so we tested it:

- We **removed the least reliable brain region** and ran everything again. The same three
  paths came back.
- **95% of people stayed on the same path** in both runs, and their stages barely changed.

### 4. An interesting hint (not yet a conclusion)

On average, patients on the **parietal-first** path did worst on a memory and thinking test,
and those on the **temporal-first** path did best, even though the doctors rated them as
similarly ill. This was the same in both runs. But with only about 25 patients per path,
it could still be chance, so it is a question for a bigger study, not a finding.

## Checking our own work

A large part of this project was making sure the results could be trusted. Three problems
were caught and fixed along the way:

1. **Faulty scans.** The first version of the processed scans made healthy brains look as if
   they *gained* tissue with age, which is impossible. We caught this with sanity checks
   before trusting any results, and switched to a differently processed version of the same
   scans that passed every check.
2. **A filter that removed the wrong people.** Our first quality filter, meant to remove bad
   scans, was mostly removing the patients with the most severe damage. We replaced it with
   one that looks only at scan quality.
3. **Mixed-up labels.** The software was sometimes attaching people to the wrong path, so the
   pictures and the tables did not match. We found this by comparing two runs, and added an
   automatic check so it cannot happen silently again.

## Limits

- **One scan per person.** The paths are worked out from snapshots, not watched happening.
  Following the same people over time would be the real test.
- **A small group** of 190 people from a single scanner. Bigger and more varied studies are
  needed before drawing firm conclusions.
- **Not a diagnostic tool.** It describes patterns across a group; it does not tell any
  individual what is happening in their brain.

## Learn more

- **[Technical write-up](TECHNICAL.md):** methods, statistics, design decisions and how to
  run the code.
- **Data:** [OASIS-1](https://sites.wustl.edu/oasisbrains/), Open Access Series of Imaging
  Studies (Marcus et al., 2007).
- **Method:** SuStaIn, Young et al., *Nature Communications*, 2018.
