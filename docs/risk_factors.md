# Risk factor table (issue #35)

**Status:** draft for team review. Needs sign-off from 2 teammates before #33 (rules engine) builds on it.

CUAD has no risk labels, so this table is the team's own definition of risk. It drives the rules engine
(#33), the Low/Medium/High cutoffs (#34), the annotation guide for the team validation set (#36) and the
validation in #37.

## Assumptions

- **Whose risk?** CUAD labels say a clause exists, not which party it favours. A rule can't tell whether
  "our" side is capped or uncapped. So the score means **"how much this clause needs a reviewer's
  attention"** for the contract as a whole, not risk to one specific party. Where wording shows the clause
  is one-sided (e.g. only one party may terminate), that raises the score.
- **Found clauses are scored; missing clauses are only flagged.** A risk score comes from a clause being
  there. When a clause whose absence matters isn't found, the report shows a **"Not found — check"** flag
  instead of adding risk, because "not found" may mean the model missed it (see Limitations). Absence is
  flagged only where a source says it matters.
- **Weights are 1–5** (5 = most serious). A rule scores a detected clause by starting from its weight and
  adjusting up or down using the "riskier when" / "less risky when" signals.
- **CUAD's `Group` codes are metadata, not risk levels** (see `data/cuad/category_descriptions.csv`).
- **"% of contracts"** is how often the clause appears across all 510 CUAD contracts (`CUADv1.json`). It
  shows how often a rule would fire on a found clause (or how often a "Not found — check" flag would appear).

## Risk categories (13 + 1 paired)

| Category | Weight | % of contracts | Why it matters | Riskier when (text signals) | Less risky when | If not found | Sources |
|---|---|---|---|---|---|---|---|
| **Uncapped Liability** | 5 | 21.8% | A party's exposure for some breaches has no ceiling. Carve-outs from a cap (e.g. IP infringement, confidentiality, data breach) often decide the real exposure more than the cap amount does. | Broad carve-outs: indemnification, confidentiality, IP infringement, data breach; words like "unlimited", "shall not apply to" | Carve-outs limited to fraud, gross negligence or wilful misconduct; a "super cap" instead of no cap | No flag | S1, S3 |
| **Cap on Liability** | — (flag only) | 53.9% | A cap is protective. Its *absence* means no contractual limit on damages, which reviewers treat as a trap worth flagging. | Cap present but no amount or formula found; many exclusions from the cap; service credits count toward the cap | Clear amount or formula (e.g. "fees paid in the prior 12 months") | **Flag:** "No limitation of liability found — check." (Missing caps are a common review issue [S4].) | S1, S3, S4 |
| **Change of Control** | 4 | 23.7% | Lets a counterparty terminate or demand consent if a party is acquired, which can affect a deal's value and timeline. | Termination right at "sole discretion" or "adverse to its interests"; no time window to exercise it; "indirect" ownership triggers; clawback or accelerated payment | Notice only; consent "not to be unreasonably withheld"; fixed exercise window | No flag (anti-assignment can still apply) | S1, S7, S8 |
| **Anti-Assignment** | 2 | 73.3% | Requires consent or notice before the contract is transferred. Matters most in asset sales, where contracts move by assignment. | Covers transfers "by operation of law" or by merger; consent at the counterparty's discretion | Consent "not to be unreasonably withheld"; exceptions for affiliates or successors | No flag | S1, S6, S7, S8 |
| **Non-Compete** | 4 | 23.3% | Restricts a party from competing in a business, geography or sector. Enforceability varies a lot by state (e.g. largely barred in California). | Broad scope (all similar business lines); long duration; wide or worldwide geography; continues after termination | Narrow scope tied to the contract's subject; short duration | No flag | S1, S10 |
| **Exclusivity** | 4 | 35.3% | An exclusive-dealing commitment can lock a party out of a whole market. It can be buried in definitions or the "Entire Agreement" clause. | "exclusive", "sole", "all requirements"; one-way; survives termination; combined with a unilateral renewal | Mutual; limited territory, product or time | No flag | S1, S4 |
| **Termination for Convenience** | 3 | 35.9% | One or both parties can end the contract without cause, trading certainty for flexibility. | Only one party may terminate; short notice; no termination fee; unclear treatment of prepaid amounts | Mutual right; long notice; minimum term; termination fee | No flag | S1, S3, S5 |
| **Renewal Term** | 3 | 34.5% | Automatic renewal can lock a party into another term if a notice deadline is missed. | "automatically renew", "successive"; renewal right held by one party only (unilateral) | Renewal requires both parties' agreement | No flag (expiry without renewal can disrupt service, but isn't scored) | S1, S3, S4 |
| ↳ **Notice Period to Terminate Renewal** | 2 | 21.8% | Sets the deadline for stopping a renewal. The longer it is, the easier it is to miss. | 90+ days | 30 days or less | **Flag when Renewal Term is found but no notice period:** "Renewal found but no notice period — check how renewal is stopped." | S1, S3 |
| **IP Ownership Assignment** | 4 | 24.3% | IP created by one party becomes the other's property. Broad definitions of "deliverables" or "work product" can capture a party's own tools and improvements. | "all right, title and interest"; "work made for hire"; broad definitions covering improvements or background IP | Limited to deliverables built from materials the other party supplied | No flag | S1, S4 |
| **Most Favored Nation** | 3 | 5.5% | Promises a party terms at least as good as anyone else's. Vague wording invites breach and disputes, and MFNs can draw antitrust scrutiny. | Short, open-ended wording ("no greater than prices charged to any other customer"); no defined comparison group | Specifies exactly which products and which customer arrangements are compared | No flag | S1, S4 |
| **Liquidated Damages** | 3 | 12.0% | A fixed amount or formula owed on breach or termination (including termination fees). Courts won't enforce one that works as a penalty. | Large fixed amounts; payable for minor breaches; reads as punitive | Tied to a reasonable formula; capped | No flag | S1, S9 |
| **Insurance** | 2 | 32.5% | Requires one party to carry insurance for the other's benefit: a cost obligation, and part of how risk is allocated. | High required coverage amounts; "additional insured" requirements | Standard coverage types and amounts | Undecided whether to flag: see open questions | S1 only, **needs a source** |
| **Audit Rights** | 2 | 42.0% | Lets a party inspect the other's books, records or premises. Unbounded audits are costly and can be used as leverage in disputes. | "at any time"; "without prior notice"; no frequency or scope limit | Once per year; 30 days' notice; defined scope | No flag | S1, S4 |

## How the weights were chosen

No source assigns numeric weights. The guides explain *why* each clause matters, and the CUAD paper
doesn't rank categories. The weights are the team's judgment, ordered roughly by **how bad the worst case
is and how hard it is to get out of**:

| Weight | Reasoning | Categories |
|---|---|---|
| 5 | Money at risk with no upper limit | Uncapped Liability |
| 4 | Could cost a lot or restricts the business itself, and is hard to undo once signed | Change of Control, Non-Compete, Exclusivity, IP Ownership Assignment |
| 3 | Real cost or lock-in, but bounded or can be planned around | Termination for Convenience, Renewal Term, Most Favored Nation, Liquidated Damages |
| 2 | Mainly an obligation or admin burden, or so common that a higher weight would flag most contracts | Notice Period to Terminate Renewal, Insurance, Audit Rights, Anti-Assignment |

Cap on Liability has no weight: a cap is protective, and its absence produces a "Not found — check" flag
instead of a score.

The weights are a starting proposal, not a finding. The team validation set (#36) scores clauses by hand
without seeing them, and #37 measures how well the rules agree. Weights should be revised if the
annotators consistently rank a category higher or lower than its weight suggests.

## Not risk-scored

These categories are still predicted by the model and shown in the report, but get no risk score:

- **Basic information:** Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Governing Law.
- **Licensing details:** License Grant, Non-Transferable License, Affiliate License-Licensor/Licensee,
  Unlimited/All-You-Can-Eat-License, Irrevocable or Perpetual License, Source Code Escrow.
- **Other commercial terms:** No-Solicit of Customers, Competitive Restriction Exception, No-Solicit of
  Employees, Non-Disparagement, Rofr/Rofo/Rofn, Revenue/Profit Sharing, Price Restrictions, Minimum
  Commitment, Volume Restriction, Joint IP Ownership, Post-Termination Services, Warranty Duration,
  Covenant Not to Sue, Third Party Beneficiary.

Candidates to promote later if the team wants wider coverage: Minimum Commitment, Post-Termination
Services, No-Solicit of Customers, Covenant Not to Sue.

## Reporting with the 12-group mapping

**Proposal:** use Arielle's 12 groups (`notebooks/cuad_grouping`) only to organise the contract report
(e.g. "Liability: High"). Keep all scoring at the 41-category level, which is what the model predicts. Team to decide.

## Limitations

- **Nobody on the team is a lawyer.** The weights and "riskier when" signals come from public practitioner
  guides plus team judgment, not legal expertise. Outside review (e.g. the AI Studio coach, a law-school
  contact, or the Atticus Project labeling handbook) would strengthen them.
- **"Not found" isn't the same as "missing".** The best model still misses roughly 1 in 5 real clauses on
  validation, so a "Not found — check" flag may be a model miss. That's why absence is flagged for a
  reviewer to confirm, not scored.
- **The tool is for triage, not legal judgment.** Its job is to point a reviewer to the right clauses, with a
  reason and a source for each flag, so they can decide which contracts to open first. It doesn't tell them
  what the legal consequences are.
- **Team validation measures consistency, not correctness.** The #36 annotators aren't lawyers either.
  Agreement between the rules and the team shows the rules match the team's shared judgment, not that
  they're legally right. Final results should state this.

## Open questions for reviewers

1. Are the weights right? (Anti-Assignment is set to 2 rather than 3 because it appears in 73% of
   contracts and would otherwise add risk to most of them.)
2. Insurance: should a *missing* insurance clause be flagged "Not found — check"? Needs a source either way.
3. Should any "not risk-scored" category be promoted (see candidates above)?
4. Agree to use the 12 groups for reporting only?

## Sources

- **S1** — CUAD category descriptions, `data/cuad/category_descriptions.csv` (The Atticus Project).
- **S2** — Hendrycks, Burns, Chen & Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract
  Review", arXiv:2103.06268 (2021). Background for the whole table: the 41 categories are ones lawyers
  "pay particular attention to" in review, and judging risk is "highly contextual" (depends on industry
  and risk tolerance). The paper doesn't rank categories or treat missing clauses as risks.
- **S3** — A. Benjamin Klaber & Aaron J. Rogers, "A (Pro)Curated List of Contractual Provisions Sourcing
  Teams Should Not Ignore (Part 1)", Morgan Lewis, 22 Jul 2026.
  https://www.morganlewis.com/blogs/sourcingatmorganlewis/2026/07/a-pro-curated-list-of-contractual-provisions-sourcing-teams-should-not-ignore-part-1
- **S4** — Alexander J. Civetta, "Six Common Contract Traps: A Field Guide", Mintz, Mar 2022 (updated 2026).
  https://www.mintz.com/insights-center/viewpoints/2866/2022-03-24-six-common-contract-traps-field-guide
- **S5** — Aaron J. Rogers & A. Benjamin Klaber, "Termination for Convenience: The Battle over Free Will",
  Morgan Lewis, 8 Jul 2026.
  https://www.morganlewis.com/blogs/sourcingatmorganlewis/2026/07/termination-for-convenience-the-battle-over-free-will
- **S6** — Glenn D. West, "Mergers and Restrictions on Assignments by 'Operation of Law'", Weil, Gotshal &
  Manges, 22 Sep 2020. https://privateequity.weil.com/?p=7227
- **S7** — "Change of Control?", Sidley Austin, Jul 2020.
  https://www.sidley.com/en/insights/publications/2020/07/change-of-control
- **S8** — Laurent Campo, "The Change of Control Problem Nobody Owns in M&A Until It's Too Late", Potomac
  Law Group, 19 Feb 2026.
  https://www.potomaclaw.com/news-Change-of-Control-Problem-Nobody-Owns-in-M-and-A-Until-Its-Too-Late
- **S9** — "Liquidated damages", Wex, Cornell Legal Information Institute.
  https://www.law.cornell.edu/wex/liquidated_damages
- **S10** — "Covenant not to compete", Wex, Cornell Legal Information Institute.
  https://www.law.cornell.edu/wex/covenant_not_to_compete
