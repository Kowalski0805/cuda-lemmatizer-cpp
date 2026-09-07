// chapter3.js — regenerates Chapter 3 (Methodology) with the three-factor cost
// model. Run: node chapter3.js  -> chapter3_methodology_v2.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  TabStopType, Math: M, MathRun, MathFraction, MathSubScript, MathSuperScript,
  MathSubSuperScript, MathSum, MathRoundBrackets, Table, TableRow, TableCell,
  WidthType, BorderStyle,
} = require("docx");

const FONT = "Times New Roman";
const SZ = 28;        // 14 pt
const LINE = 360;     // 1.5 spacing
const RIGHT_TAB = 9070;

const mr = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });
const sup = (b, s) => new MathSuperScript({ children: [mr(b)], superScript: [mr(s)] });
const frac = (num, den) => new MathFraction({ numerator: num, denominator: den });

// Body paragraph.
const p = (text, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.BOTH,
    spacing: { line: LINE, after: 120 },
    indent: { firstLine: opts.noIndent ? 0 : 567 },
    children: [new TextRun({ text, font: FONT, size: SZ, ...opts.run })],
  });

// Paragraph built from mixed runs: strings and {i:"..."} / {b:"..."} objects.
const rich = (parts, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.BOTH,
    spacing: { line: LINE, after: 120 },
    indent: { firstLine: opts.noIndent ? 0 : 567 },
    children: parts.map((x) =>
      typeof x === "string"
        ? new TextRun({ text: x, font: FONT, size: SZ })
        : new TextRun({
            text: x.i || x.b,
            font: FONT,
            size: SZ,
            italics: !!x.i,
            bold: !!x.b,
          })
    ),
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 180, line: LINE },
    children: [new TextRun({ text, font: FONT, size: 32, bold: true })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 140, line: LINE },
    children: [new TextRun({ text, font: FONT, size: SZ, bold: true })],
  });

// Displayed, right-numbered equation: tab -> centred math -> tab -> "(n)".
const eq = (children, num) =>
  new Paragraph({
    spacing: { before: 140, after: 140, line: LINE },
    tabStops: [
      { type: TabStopType.CENTER, position: Math.floor(RIGHT_TAB / 2) },
      { type: TabStopType.RIGHT, position: RIGHT_TAB },
    ],
    children: [
      new TextRun({ text: "\t", font: FONT, size: SZ }),
      new M({ children }),
      new TextRun({ text: `\t(${num})`, font: FONT, size: SZ }),
    ],
  });

// Compact data table.
const cell = (text, { bold = false, align = AlignmentType.CENTER } = {}) =>
  new TableCell({
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [
      new Paragraph({
        alignment: align,
        spacing: { line: 240 },
        children: [new TextRun({ text, font: FONT, size: 24, bold })],
      }),
    ],
  });

const table = (rows) =>
  new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 4 },
      left: { style: BorderStyle.NONE },
      right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2 },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: rows.map(
      (r, i) =>
        new TableRow({
          children: r.map((c, j) =>
            cell(c, { bold: i === 0, align: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })
          ),
          tableHeader: i === 0,
        })
    ),
  });

const caption = (text) =>
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 80, after: 200, line: 240 },
    children: [new TextRun({ text, font: FONT, size: 24, italics: true })],
  });

const body = [];
const P = (...a) => body.push(p(...a));
const R = (...a) => body.push(rich(...a));
const H1 = (t) => body.push(h1(t));
const H2 = (t) => body.push(h2(t));
const EQ = (...a) => body.push(eq(...a));
const T = (rows, cap) => { body.push(table(rows)); body.push(caption(cap)); };

// ============================================================ 3
H1("3. Methodology");

P(
  "This chapter formalizes the input-ordering problem for GPU-resident trie traversal, derives a cost model that predicts when reordering is profitable, and specifies the ordering algorithms compared in the experimental study. The workload throughout is dictionary lemmatization of a token stream by traversal of a byte-level trie held in GPU memory: each thread processes one token, descending one trie level per input byte. Prior profiling established that the traversal kernel is latency-bound rather than compute-bound, with the transition-fetch loop dominating warp stalls; input ordering is therefore studied as a mechanism acting on memory-system behavior, not on arithmetic work, which is identical in every variant.",
  { noIndent: true }
);

P(
  "The model developed in Section 3.3 differs from the one assumed at the outset of this work in a way that the measurements forced. Reordering was initially treated as acting on a single quantity — the locality of trie-state accesses. It in fact acts on three distinct quantities, two of which improve with ordering precision while the third degrades, and the degrading one dominates unless a further step is taken. The cost model is therefore presented as a factorization whose terms have disjoint dependencies, since it is that separability, rather than any single term, that makes the optimum predictable."
);

// ============================================================ 3.1
H2("3.1. Notation and problem statement");

R([
  "Let ",
  { i: "W" },
  " = (",
  { i: "w" },
  "₁, …, ",
  { i: "w" },
  "ₙ) be a stream of ",
  { i: "n" },
  " tokens over the byte alphabet Σ = {1, …, 255} (UTF-8 encoded; the NUL byte cannot occur in text). ∣",
  { i: "w" },
  "∣ denotes byte length; for Ukrainian Cyrillic every letter occupies two bytes, so a depth of 2",
  { i: "c" },
  " bytes corresponds to ",
  { i: "c" },
  " letters. The trie ",
  { i: "T" },
  " has state set ",
  { i: "Q" },
  "; lookup of ",
  { i: "w" },
  " visits ∣",
  { i: "w" },
  "∣ states, and the per-level state fetch is the dominant memory operation. Threads are scheduled in warps of ",
  { i: "g" },
  " = 32.",
], { noIndent: true });

R([
  "An ordering algorithm produces a permutation π of {1, …, ",
  { i: "n" },
  "}. In every variant π exists only as a transient index array: the traversal kernel reads token π(",
  { i: "i" },
  ") and scatters its result to output position π(",
  { i: "i" },
  "), so the output vector is always in original stream order and no reordered copy of the corpus is ever retained. This eliminates the rebuild cost of maintaining a sorted corpus, makes all variants checksum-identical by construction, and confines the differences between variants to (a) the cost of computing π and (b) the memory-system behavior induced by π during traversal.",
]);

R([
  { b: "Three" },
  " mechanisms link ordering to traversal speed, and they do not act in the same direction. ",
  { b: "Path sharing" },
  ": if the ",
  { i: "g" },
  " tokens of a warp share prefixes, their descents visit the same trie states, and per-level state fetches collapse into few cache lines. ",
  { b: "Trip-count uniformity" },
  ": a warp executes until its longest token terminates, so mixing lengths within a warp idles the threads holding short tokens. ",
  { b: "Stream coalescing" },
  ": in natural order, thread ",
  { i: "i" },
  " and thread ",
  { i: "i" },
  "+1 read adjacent bytes of the input buffer, so a warp's token-byte reads are served by a few contiguous sectors — a property the unordered stream possesses for free and that ",
  { i: "any" },
  " permutation destroys, since a permuted read gathers from arbitrary offsets.",
]);

R([
  "Alphabetical ordering targets the first mechanism, length ordering the second; both sacrifice the third. This is the central tension of the chapter: reordering does not add structure to an unstructured stream, it ",
  { i: "exchanges" },
  " one kind of structure for another, and the exchange is not obviously favorable. Section 3.4 introduces an algorithm class that recovers the third mechanism by re-materializing the tokens in permuted order, at a cost that is itself accounted for in the model.",
]);

// ============================================================ 3.2
H2("3.2. Order-preserving key construction");

R([
  "All GPU variants reduce string comparison to integer comparison by packing a token prefix into a 64-bit key big-endian, i.e. with earlier bytes in more significant positions. With ",
  { i: "b" },
  "ⱼ ∈ Σ the ",
  { i: "j" },
  "-th byte of ",
  { i: "w" },
  ":",
], { noIndent: true });

EQ(
  [
    mr("φ(w) = "),
    new MathSum({
      children: [mr("b"), mr("ⱼ"), mr(" · "), sup("256", "7−j")],
      subScript: [mr("j=0")],
      superScript: [mr("m−1")],
    }),
    mr(",   m = min(∣w∣, 8)."),
  ],
  1
);

R([
  { b: "Proposition 1." },
  " For any two tokens ",
  { i: "u" },
  ", ",
  { i: "v" },
  ": φ(",
  { i: "u" },
  ") < φ(",
  { i: "v" },
  ") if and only if ",
  { i: "u" },
  " precedes ",
  { i: "v" },
  " in bytewise lexicographic order truncated to 8 bytes. ",
  { i: "Proof sketch." },
  " Positional weights 256^(7−j) make the first differing byte within the covered window dominate all subsequent bytes, exactly as in lexicographic comparison; absent bytes contribute 0 and, since NUL is excluded from Σ, a proper prefix receives a strictly smaller key than any extension of it. ∎",
]);

P(
  "Composite keys express the length-based orderings in the same 64-bit format, which lets a single radix-sort machinery serve all three comparison axes: the length-only key places the token length in the top byte, and the length-then-alphabetical key concatenates it with the first seven bytes of the token:"
);

EQ(
  [
    sub("κ", "len"),
    mr("(w) = ∣w∣ · "),
    sup("2", "56"),
    mr(",    "),
    sub("κ", "la"),
    mr("(w) = ∣w∣ · "),
    sup("2", "56"),
    mr(" + ⌊φ(w) / "),
    sup("2", "8"),
    mr("⌋."),
  ],
  2
);

P(
  "A remark on collation: raw byte order over UTF-8 does not coincide with Ukrainian alphabetical order for the letters і, ї, є and ґ, whose code points fall outside the contiguous а–я block. This is immaterial here. The locality mechanism requires only that tokens with equal prefixes become adjacent, a property of any prefix-consistent total order; linguistic collation buys nothing additional. Byte order is therefore used throughout, and «alphabetical» below means bytewise lexicographic."
);

// ============================================================ 3.3
H2("3.3. Cost model: a decomposition, and its correction by measurement");

R([
  "The traversal kernel is latency-bound, and the measurements of this study confirm it independently: in natural order the kernel moves 23.5 bytes of DRAM traffic per token while sustaining roughly 0.53 ns per token, which corresponds to a delivered bandwidth two orders of magnitude below the device peak. Per-token time is therefore modelled as a product of the number of memory requests a token generates, the effective cost of a request, and the fraction of a warp's execution that does useful work:",
], { noIndent: true });

EQ(
  [
    mr("τ(π, n) ≈ "),
    frac([mr("ν(π) · "), sub("L", "eff"), mr("(h(π, n))")], [mr("e(π)")]),
  ],
  3
);

R([
  "The three factors are measured separately, and the substance of the model is that they depend on ",
  { i: "different" },
  " variables.",
]);

R([
  { b: "Requests per token, ν." },
  " Two streams are read during traversal: the trie states and the token bytes themselves. Their request counts add,",
]);

EQ([mr("ν(π) = "), sub("ν", "T"), mr("(π) + "), sub("ν", "S"), mr("(π),")], 4);

R([
  "and the second term is where ordering does damage. Reading tokens in stream order, a warp's ",
  { i: "g" },
  " tokens occupy a contiguous span of about ",
  { i: "g" },
  "·⟨∣w∣⟩ bytes, served by ⌈",
  { i: "g" },
  "·⟨∣w∣⟩ / 32⌉ sectors, so the per-token cost is a small fraction of a sector. Reading them through a permutation, each token is an independent gather:",
]);

EQ(
  [
    sub("ν", "S"),
    mr("(id) ≈ ⟨∣w∣⟩ / 32,     "),
    sub("ν", "S"),
    mr("(π) ≈ ⌈(⟨∣w∣⟩ + 31) / 32⌉ ≈ 2   for π ≠ id."),
  ],
  5
);

R([
  "For the corpora used here (⟨∣w∣⟩ ≈ 12 bytes) this is a difference between roughly 12 and 64 bytes of DRAM traffic per token — the measured values are 23.5 B/token in natural order against 79.0 B/token under an 8-byte sort, the residual in each case being the trie stream. A permutation thus multiplies total traffic by about 3.4 before it has improved anything.",
]);

R([
  { b: "Cost per request, " },
  { b: "L" },
  { b: "_eff." },
  " With ",
  { i: "h" },
  " the cache hit fraction,",
]);

EQ(
  [
    sub("L", "eff"),
    mr(" = h · "),
    sub("L", "cache"),
    mr(" + (1 − h) · "),
    sub("L", "DRAM"),
    mr("."),
  ],
  6
);

R([
  "Ordering acts on ",
  { i: "h" },
  " through intra-warp coincidence (a shared path is fetched once and broadcast) and inter-warp temporal reuse (a compact working set of hot subtries survives in L2). The top trie levels are cache-resident under any ordering — the root of the trie used here has only four transitions, one per Cyrillic lead byte — so the attainable gain is concentrated at middle depths, where the number of live subtries is large enough to thrash the cache under random order yet small enough to fit when accesses are grouped.",
]);

R([
  { b: "SIMT efficiency, " },
  { b: "e" },
  ". A warp retires only when its longest token does, so for lengths ",
  { i: "L" },
  "₁, …, ",
  { i: "L" },
  "_g drawn from the corpus length distribution",
]);

EQ(
  [
    mr("e = "),
    frac([mr("E[L]")], [mr("E[max("), sub("L", "1"), mr(", …, "), sub("L", "g"), mr(")]")]),
  ],
  7
);

R([
  "which for a random warp composition is substantially below one, since the expected maximum of 32 draws from a right-skewed length distribution far exceeds its mean, and approaches one when the stream is length-sorted.",
]);

R([
  { b: "Proposition 2 (separability)." },
  " ν and ",
  { i: "e" },
  " depend on π only through the ordering precision and are invariant in ",
  { i: "n" },
  "; ",
  { i: "h" },
  " depends on ",
  { i: "n" },
  " through the size of the working set. ",
  { i: "Evidence." },
  " Seven algorithms were profiled at seven corpus sizes spanning 1 M to 50 M tokens — forty-nine measurements — with the transaction count taken directly from the sector counter rather than inferred. Across that fiftyfold range ν varies by a mean of 2.1 % per algorithm, and by no more than 4.3 % for any of them (natural order 56.7 → 53.6 transactions per token; 8-byte sort 19.7 → 18.7; sorted-and-compacted 12.9 → 12.0). Warp efficiency varies by 1.6 %. Over the same range the L2 hit fraction falls by about 10 points for every ordering that permutes without re-materializing (89.6 → 79.4 for the 8-byte sort, 92.0 → 80.7 for the 4-byte prefix sort) and ",
  { i: "rises" },
  " for the compacted variants (83.1 → 89.3), while remaining flat in natural order (94.1 → 95.2). ∎",
]);

R([
  "The separation is therefore sharper than the proposition asserts. It is not merely that two factors are precision-dependent and one scale-dependent: ν is flat to about two per cent for every algorithm over the whole range tested, so ",
  { i: "the entire scale dependence of the problem is carried by a single scalar per algorithm" },
  ". Which scalar that is, however, the next paragraphs revise.",
]);

R([
  "Separability is what allows the optimum to move at all. Two of the three factors can be measured once, at whatever corpus size is convenient, and carried to any other size; only the third need be re-estimated as the working set grows. It also explains a fact that is otherwise puzzling — that the ",
  { i: "optimal ordering precision falls as the corpus grows" },
  " — since precision buys ν and ",
  { i: "e" },
  " at a fixed rate while the third factor deteriorates with scale.",
]);

R([
  { b: "The model as stated does not survive its own validation." },
  " Equation (3) was fitted against the forty-nine measurements by leave-one-scale-out: each fit sees six corpus sizes and predicts the seventh, so every reported error is on an operating point the fit never saw. It scores 34.5 % mean relative error, and identifies the fastest algorithm in only five of seven folds. Three observations locate the fault in the functional form rather than in the measurements. Accuracy ",
  { i: "improves" },
  " when the factors are degraded — 33.1 % with ν and ",
  { i: "e" },
  " measured directly, 23.5 % with cruder proxies substituted for both, 28.8 % with the divergence factor deleted outright. The fitted cache latency is negative under several fit points. And the residual for the baseline is a near-constant offset of 41 to 44 % at every scale rather than a drift, which is the signature of a model that describes how cost varies but not what it is.",
]);

R([
  "The reason is visible once the right counter is collected. Equation (6) reaches for the cost of a memory access through a hit-rate-weighted average of two latencies. That average cannot represent the quantity that actually governs a latency-bound kernel, because it says nothing about how much of the latency is ",
  { i: "hidden" },
  ": two kernels with identical hit rates differ arbitrarily in runtime according to how many independent accesses they keep in flight. The exposed fraction is directly measurable as the number of warp-cycles stalled on an outstanding memory operation per cycle in which the scheduler could otherwise issue. Writing that quantity σ(π, ",
  { i: "n" },
  ") and substituting it for the whole L_eff construction gives",
]);

EQ(
  [
    mr("τ(π, n) ≈ α · ν(π) · σ(π, n) + β,"),
  ],
  "3′"
);

R([
  "with α and β the only free parameters. On the same folds this scores ",
  { b: "13.1 % overall, and 15.0 % on the folds whose measurements resolve it, against Eq. (3)'s 34.5 %, with one parameter fewer" },
  ", and it absorbs the divergence factor rather than competing with it: dividing Eq. (3′) by ",
  { i: "e" },
  " makes the fit worse (23.9 %), so warp efficiency is not an independent term of the cost but one of the mechanisms by which a warp comes to expose latency. The cache hit rates likewise cease to earn their place once σ is present; adding an explicit DRAM term to Eq. (3′) improves it only from 13.1 % to 11.6 %, which is smaller than the fold-to-fold spread of Eq. (3′) itself and is declined accordingly — Section 4.9 gives the resolution argument.",
]);

R([
  "Proposition 2 holds for Eq. (3′) in the form given above, and more cleanly: ν is the precision-dependent factor and σ the scale-dependent one. The scale sensitivity of σ recovers the experimental ranking of the algorithms with no reference to any timing — it varies by 1.7 % across the range for natural order and 7.8 % for coarse ordering with compaction, the two strategies that prove scale-invariant in Chapter 4, and by 35 to 43 % for the fine orderings, which are precisely the ones that decay.",
]);

R([
  { b: "What Eq. (3′) does not do." },
  " σ is obtained from the kernel it describes, so the model explains and interpolates but does not predict in advance: choosing an algorithm for an unmeasured corpus size still requires modelling σ itself from ordering precision and working-set size. The causal chain is scale → L2 residency → latency exposure → time, and this study measures its endpoints but not its middle link. Nor should the model be used to resolve close comparisons: the two compacted variants differ by less than 20 % beyond 35 M tokens, and which of them a fitted model selects there is not stable across model families. Equation (3′) is offered as a decomposition that survives quantitative test, not as a calculator.",
]);

R([
  { b: "Amortization." },
  " Reordering is worthwhile only if its cost is recovered. With ",
  { i: "C" },
  "_prep(",
  { i: "A" },
  ") the cost of computing π under algorithm ",
  { i: "A" },
  ", τ₀ and τ_A the per-token traversal times without and with the ordering, and ",
  { i: "k" },
  " the number of traversals performed over the same permutation,",
]);

EQ(
  [
    sub("C", "prep"),
    mr("(A) < k · n · ("),
    sub("τ", "0"),
    mr(" − "),
    sub("τ", "A"),
    mr(")."),
  ],
  8
);

R([
  "The case ",
  { i: "k" },
  " = 1 — a single pass over freshly arrived data, which is the common deployment — is a strictly stronger requirement than the reuse case and is treated separately throughout the results, because the two regimes select different algorithms. Since ",
  { i: "C" },
  "_prep grows linearly in ",
  { i: "n" },
  " while the gain per token shrinks with ",
  { i: "n" },
  " by Proposition 2, criterion (8) at ",
  { i: "k" },
  " = 1 is satisfied only inside a bounded interval of corpus sizes; locating that interval is one of the principal experimental results.",
]);

// ============================================================ 3.4
H2("3.4. The ordering algorithms");

R([
  "Nine algorithms are compared, labeled A0–A6b and corresponding one-to-one to the variants of the benchmark implementation. All GPU sorting builds on least-significant-digit radix sort with ",
  { i: "r" },
  "-bit digits (",
  { i: "r" },
  " = 8 in the implementation, via CUB), whose cost is linear in the number of digit passes:",
], { noIndent: true });

EQ(
  [
    mr("P(X) = ⌈8X / r⌉,     "),
    sub("C", "radix"),
    mr("(X) ≈ P(X) · "),
    sub("c", "pass"),
    mr(" · n,"),
  ],
  9
);

R([
  "where ",
  { i: "X" },
  " is the number of key bytes participating in the sort. Restricting the sorted bit range is therefore a direct dial between ordering precision and preparation cost.",
]);

R([{ b: "A0 (baseline)." }, " Natural stream order, π = id. Defines τ₀ and the reference checksum against which every other variant is verified. By Eq. (5) it is also the only variant that enjoys full stream coalescing without paying for it."]);

R([{ b: "A1 (CPU comparison sort)." }, " An index array is sorted on the host by the key of Eq. (1)/(2), with ties beyond the 8-byte window broken by full bytewise comparison, yielding an exact order at Θ(n log n) comparisons. Only the 4n-byte permutation crosses the PCIe bus. This is the classical option; the study charges its cost fully to preparation, giving the most conservative placement in criterion (8)."]);

R([{ b: "A2 (GPU radix sort, 8-byte key)." }, " CUB radix sort over the full key: P(8) = 8 passes, exact to a depth of 8 bytes — four Cyrillic letters. Tokens identical in their first four letters retain arbitrary relative order."]);

R([
  { b: "A2b (exact full-depth sort by segmented refinement)." },
  " A2 is completed to exact lexicographic order without re-sorting the whole array. After the initial pass, maximal runs of equal keys are identified by head flags and an inclusive scan; for refinement depth ",
  { i: "d" },
  " ≥ 1 the next 8-byte window of each token is packed as in Eq. (1), and each tie run is sorted independently with a segmented radix sort. Head flags only accumulate, so a later pass cannot reorder across a boundary established earlier; by induction the order after pass ",
  { i: "d" },
  " is exact to depth 8(",
  { i: "d" },
  "+1) bytes. Since NUL bytes cannot occur, the depth-",
  { i: "d" },
  " key vanishes if and only if the token is exhausted; a tie run whose keys are all zero therefore consists of identical tokens and is excluded from refinement — the decisive detail under Zipf-distributed streams, where repeated tokens form enormous runs that no pass can resolve. Its cost is",
]);

EQ(
  [
    sub("C", "A2b"),
    mr(" = "),
    sub("C", "radix"),
    mr("(8) + "),
    new MathSum({
      children: [sub("c", "seg"), mr("("), sub("m", "d"), mr(")")],
      subScript: [mr("d ≥ 1")],
    }),
    mr(","),
  ],
  10
);

R([
  "where the tie mass ",
  { i: "m" },
  "_d counts positions inside active tie runs at depth ",
  { i: "d" },
  ". A2b prices exact ordering honestly: if its traversal time equals A2's, exactness is demonstrably not worth even this increment.",
]);

R([{ b: "A3 (prefix-restricted sort)." }, " The radix sort of A2 confined to the top X key bytes by restricting the digit range: P(X) passes, ordering exact to X bytes and arbitrary beyond. Sweeping X ∈ {2, 4, 8} traces the precision–cost dial explicitly."]);

R([
  { b: "A4 (single-pass partition)." },
  " Exactly one counting pass: a histogram over the top β key bits, an exclusive scan of the 2^β bin counts, and a scatter of indices to bin-contiguous positions. The result is a partial order — equivalence classes on the top β bits, arbitrary order within a class — at Θ(",
  { i: "n" },
  ") cost with only two passes touching the data. With β = 16, one class corresponds to one leading Cyrillic letter.",
]);

R([
  { b: "A5 (streamed partition pipeline)." },
  " A4 embedded in a real-time regime, where the stream arrives in batches of ",
  { i: "B" },
  " tokens and a monolithic sort of the whole input is unavailable by definition. A copy stream performs transfer, key generation and partition of batch ",
  { i: "k" },
  "+1 while ",
  { i: "S" },
  " worker streams traverse batch ",
  { i: "k" },
  "; double buffering with device-side event dependencies keeps the host out of the loop. Steady-state throughput is governed by the slowest stage,",
]);

EQ(
  [
    sub("R", "pipe"),
    mr(" = "),
    frac(
      [mr("B")],
      [mr("max("), sub("T", "copy"), mr(", "), sub("T", "prep"), mr(", "), sub("T", "trav"), mr(")")]
    ),
    mr(","),
  ],
  12
);

R([
  "and the bin space is partitioned statically across the worker streams, so stream ",
  { i: "s" },
  " sees the same leading-letter range in every batch and accumulates temporal reuse of the corresponding subtries. Because Ukrainian leading-letter frequencies are far from uniform, the static assignment is load-imbalanced; with ",
  { i: "n" },
  "_s the tokens routed to stream ",
  { i: "s" },
  ", the efficiency of the worker phase is",
]);

EQ(
  [
    sub("η", "bal"),
    mr(" = "),
    frac(
      [mr("(1 / S) · "), new MathSum({ children: [sub("n", "s")], subScript: [mr("s")] })],
      [mr("max"), mr("ₛ"), mr(" "), sub("n", "s")]
    ),
    mr("."),
  ],
  13
);

R([
  "It should be stressed that on a single device the L2 cache is shared by all streaming multiprocessors, so A5 does not give each kernel a private cache. What the static assignment buys is copy/compute overlap and a ",
  { i: "partitioned working set" },
  ": concurrently resident kernels touch disjoint subtries rather than evicting one another's lines. The mechanism is working-set partitioning, not cache isolation, and it is reported as such.",
]);

R([
  { b: "A6 and A6b (sort with compaction)." },
  " The algorithm class that the cost model demands. A6 sorts as A2 does, then ",
  { i: "materializes the tokens in permuted order" },
  " by a gather pass, so that traversal reads them sequentially and ",
  { i: "ν" },
  "_S returns to its natural-order value while ",
  { i: "e" },
  " and ",
  { i: "h" },
  " retain the benefit of ordering. A6b is identical but sorts to 2-byte precision only. The compacted buffer is transient device memory: the corpus on disk is untouched and results still scatter to original positions, so the no-rebuild property of Section 3.1 is preserved. The additional cost is one read and one write of the token bytes plus an exclusive scan of the lengths,",
]);

EQ(
  [
    sub("C", "comp"),
    mr(" ≈ (2 · n · ⟨∣w∣⟩ + 4n) / "),
    sub("B", "dev"),
    mr(","),
  ],
  11
);

R([
  "with ",
  { i: "B" },
  "_dev the achievable device bandwidth. Because ordering acts on ",
  { i: "e" },
  " and ",
  { i: "h" },
  " while compaction acts on ν_S, the two are terms of ",
  { i: "different factors" },
  " of Eq. (3) and their effects are expected to compose multiplicatively rather than to overlap — a prediction stated in Section 3.5 and tested directly.",
]);

T(
  [
    ["Algorithm", "Precision", "Prep passes", "Extra memory", "Coalescing"],
    ["A0 baseline", "none", "0", "—", "native"],
    ["A1 CPU sort", "exact", "host Θ(n log n)", "4n", "lost"],
    ["A2 sort-8B", "8 B (4 letters)", "8", "12n", "lost"],
    ["A2b exact", "exact", "8 + refinement", "28n", "lost"],
    ["A3 prefix-X", "X B", "⌈8X / r⌉", "12n", "lost"],
    ["A4 partition", "β bits (1 letter)", "2", "8n + 2^β", "lost"],
    ["A5 streamed", "β bits, per batch", "2 per batch", "per-slot buffers", "lost"],
    ["A6 sort+compact", "8 B (4 letters)", "8 + gather", "12n + Σ∣w∣", "restored"],
    ["A6b coarse+compact", "2 B (1 letter)", "2 + gather", "12n + Σ∣w∣", "restored"],
  ],
  "Table 3.1 — The nine ordering algorithms on the four axes the cost model identifies as relevant: precision, which sets ν_T and e; preparation passes over the data, which set C_prep; device memory overhead; and whether stream coalescing survives, which sets ν_S. Only the last two rows hold all three mechanisms of Section 3.1 simultaneously."
);

// ============================================================ 3.5
H2("3.5. Predictions");

R([
  "The model of Section 3.3 makes four predictions that the experiments of the next chapter are designed to falsify.",
], { noIndent: true });

R([
  { b: "P1 (saturation)." },
  " Because the top trie levels are cache-resident under any ordering, the marginal value of ordering precision vanishes beyond the depth at which subtries stop fitting in cache. Define the saturation depth",
]);

EQ(
  [
    sup("ℓ", "∗"),
    mr(" = min { ℓ : "),
    sub("S", "ℓ"),
    mr(" > "),
    sub("S", "L2"),
    mr(" },"),
  ],
  14
);

R([
  "with ",
  { i: "S" },
  "_ℓ the aggregate size of live depth-ℓ subtries and ",
  { i: "S" },
  "_L2 the L2 capacity. The prediction is that traversal time is flat in ordering precision beyond ℓ∗, so that A3 with a small X, and A4, capture essentially the whole gain of the exact orderings A1 and A2b.",
]);

R([
  { b: "P2 (mechanism separation)." },
  " Alphabetical ordering should move the cache-hit counters through ",
  { i: "h" },
  ", length ordering the thread-uniformity counters through ",
  { i: "e" },
  ", and neither should move the other's counter family. The stronger form of the prediction, and the one that distinguishes this model from the one it replaces, is that ",
  { i: "e" },
  " alone is not worth purchasing: since length ordering achieves uniformity by destroying prefix adjacency, it raises ",
  { i: "e" },
  " while lowering ",
  { i: "h" },
  ", and the model predicts a net loss.",
]);

R([
  { b: "P3 (composition)." },
  " Ordering and compaction act on different factors of Eq. (3). Their speedups should therefore compose multiplicatively: the gain of A6 over A0 should approximate the product of the gain attributable to ",
  { i: "h" },
  " and ",
  { i: "e" },
  " (measurable as A2 over A0 once ν is held fixed) with the gain attributable to ν_S (measurable as compaction applied to an ordering that provides no path sharing at all).",
]);

R([
  { b: "P4 (scale dependence of the optimum)." },
  " By Proposition 2 only the memory factor — ",
  { i: "h" },
  " in Eq. (3), σ in Eq. (3′) — depends on ",
  { i: "n" },
  ". Since precision costs latency exposure increasingly as the working set grows, the ordering precision that minimizes τ should ",
  { i: "decrease" },
  " with corpus size, and the single-pass admissibility interval of criterion (8) should be bounded above.",
]);

// ============================================================ 3.6
H2("3.6. Measurement protocol");

R([
  "Two features of the protocol are not conventional and are stated explicitly, because in the course of this study each of them, when omitted, produced a self-consistent but entirely spurious result.",
], { noIndent: true });

R([
  { b: "Clock-state control." },
  " On a consumer device without administrative privileges the graphics and memory clocks cannot be pinned. An idle GPU drops to 210 MHz core and 405 MHz memory against maxima of 3105 and 11501 MHz. Any host-side work interposed between timed kernels — the multi-second host sort of A1 above all — therefore parks the clocks, and whichever variants are timed next measure a ramping device rather than their own locality. Measuring the variants in declaration order, one kernel each, produced a monotone ordering of results that tracked ",
  { i: "execution order" },
  " exactly and had no relation to the orderings under test. The protocol adopted instead builds every permutation first, so that no host work is interposed between timed kernels; performs a busy warm-up until clocks stabilize; and then times the traversal kernels ",
  { i: "round-robin" },
  " over several rounds, reporting per-variant medians together with the ratio of maximum to minimum as a drift indicator.",
]);

R([
  { b: "Preparation timing." },
  " Preparation cost must be measured with the same discipline as traversal. Timed once, the first construction of each ordering absorbs CUDA module loading and first-allocation costs, which are incurred once per process rather than once per batch; this inflates the apparent preparation cost and, through criterion (8), directly biases the break-even estimate that the study exists to produce. Preparation is therefore built ",
  { i: "k" },
  " times per variant with the median reported. Adopting this changed the estimated single-pass break-even corpus size by 28 % and removed a non-monotonicity from the curve.",
]);

R([
  { b: "Correctness invariant." },
  " Every variant must reproduce the baseline output bit-for-bit. Since all variants scatter through π to original positions, a single 64-bit checksum over the result vector verifies both the ordering machinery and the traversal, and any divergence is a defect rather than a trade-off. This invariant is checked on every run and holds for all nine algorithms on all corpora reported.",
]);

R([
  { b: "Counters." },
  " Per-variant hardware counters are collected for the traversal kernel at every corpus size in the sweep: global load requests and sectors, from which transactions per token and sectors per request both follow; threads retired per executed instruction, which is ",
  { i: "e" },
  " itself; L1 and L2 sector hit rates; absolute DRAM read traffic; achieved occupancy; and warp-cycles stalled on outstanding memory operations per issue-active cycle, which is σ. Two of these replace substitutes used in an earlier version of this work, and the replacement matters. Branch-target uniformity was read as warp efficiency: it reports 72.8 % for natural order where the true figure is 31.2 %, and by understating the effect of ordering threefold it understates the mechanism the study is about. Sectors per request was read as transactions per token: it omits how many requests a token issues, a quantity that differs fourfold between the permuted and the compacted algorithms, and a cost model fitted on it returns negative memory latencies. Hit rates are always reported alongside absolute traffic, since a variant that sharply reduces its request count can show a ",
  { i: "lower" },
  " hit rate while moving strictly less data — as the compacted variants do — and the ratio alone reads backwards.",
]);

R([
  { b: "Corpus statistics." },
  " Because incidental reuse of frequent tokens gives even the unordered baseline some locality, the type/token ratio and the dictionary recognition rate are reported alongside throughput for every corpus, and scale sweeps are performed on prefixes of a single corpus so that vocabulary growth follows the natural Heaps'-law trajectory rather than varying with genre.",
]);

const doc = new Document({
  creator: "",
  title: "Chapter 3. Methodology",
  styles: {
    default: {
      document: { run: { font: FONT, size: SZ } },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1134, right: 567, bottom: 1134, left: 1701 },
        },
      },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("chapter3_methodology_v2.docx", buf);
  console.log("wrote chapter3_methodology_v2.docx", buf.length, "bytes");
});
