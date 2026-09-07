// chapter3.js — regenerates Chapter 3 (Methodology) with the three-factor cost
// model. Run: node chapter3.js  -> chapter3_methodology_v2.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  TabStopType, Math: M, MathRun, MathFraction, MathSubScript, MathSuperScript,
  MathSubSuperScript, MathSum, MathRoundBrackets, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun,
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

// Centred figure at a fixed rendered width; height follows the PNG's aspect.
const FIG_W = 470;
const figure = (file, aspect) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 60, line: 240 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(`${__dirname}/figs/${file}`),
        transformation: { width: FIG_W, height: Math.round(FIG_W * aspect) },
      }),
    ],
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
const F = (file, aspect, cap) => { body.push(figure(file, aspect)); body.push(caption(cap)); };

// ============================================================ 4
H1("4. Results");

P(
  "This chapter reports the experimental comparison of the nine ordering algorithms of Section 3.4 against the production lemmatization trie. It is organized around the four predictions P1–P4 of Section 3.5: each is stated, tested, and marked confirmed or refuted. Three of the four are confirmed, one of them in a stronger form than predicted; the fourth is confirmed with a boundary condition that the prediction did not anticipate. A final section reports a clock-frequency experiment that discriminates the factors of the cost model without recourse to hardware counters, and states the threats to validity.",
  { noIndent: true }
);

// ---------------------------------------------------------- 4.1
H2("4.1. Experimental setup");

R([
  "Measurements were taken on an NVIDIA RTX 4080 SUPER (Ada, 16 GB, 64 MB L2, 320 W cap) with an AMD Ryzen 7800X3D host, CUDA 12.8 and driver 596.21. Unless stated otherwise the graphics clock is pinned at 2730 MHz and the memory clock at 11501 MHz, an operating point chosen because it reproduces the card's unpinned behavior under load (baseline 10.10 ms pinned against 10.07 ms unpinned) while removing the drift described in Section 3.6. All timings follow the protocol of that section: permutations built first, busy warm-up, traversal kernels timed round-robin over nine rounds, medians reported, preparation costs taken as the median of five builds.",
], { noIndent: true });

R([
  "The dictionary is the deployed Ukrainian morphological trie: 9 351 209 states, 9 351 208 transitions and 85.9 MB of lemma strings, 260 MB resident on the device. The traversal kernel is the deployed one — one trie level per input byte, linear scan over each state's transition list — so the measurements describe the system as shipped rather than an idealized variant.",
]);

R([
  "Two corpora are used. ",
  { i: "Fiction" },
  " (18.3 M tokens, 661 082 types, type/token 0.036, dictionary recognition 94.8 %) serves for the per-variant comparisons. ",
  { i: "Wiki" },
  " (50 M tokens, 2 172 988 types, type/token 0.044, recognition 83.8 %) serves for the scale sweeps, which are taken over prefixes of the single corpus so that vocabulary growth follows a natural Heaps'-law trajectory rather than varying with genre. Tokens are lowercased on load, matching the ICU normalization of the production pipeline.",
]);

R([
  { b: "Reproducibility." },
  " Two independent runs at the same operating point agree within 0.7 % on both medians and best-of times for every variant. The ratio of maximum to minimum round time is retained in the output as a drift alarm but is ",
  { i: "not" },
  " used as an error bar: being an extremum over nine rounds it is itself unstable, varying between 1.01 and 1.26 for the same variant across repeats.",
]);

// ---------------------------------------------------------- 4.2
H2("4.2. Correctness and the ordering ladder");

R([
  "Every variant reproduces the baseline output bit-for-bit; the 64-bit checksum over the result vector is identical for all nine algorithms on every corpus reported. The permutation-scatter construction of Section 3.1 therefore holds against the real trie, and the comparison below is between algorithms that compute the same function.",
], { noIndent: true });

T(
  [
    ["Algorithm", "Prep, ms", "Lookup, ms", "Speedup", "Mtok/s"],
    ["A0 baseline", "—", "10.07", "1.00×", "1820"],
    ["A1 CPU sort †", "5888", "10.86", "0.93×", "1687"],
    ["A2 sort-8B", "6.67", "9.93", "1.01×", "1845"],
    ["A2b exact", "103.6", "10.46", "0.96×", "1752"],
    ["A3 prefix-4B", "3.82", "8.03", "1.25×", "2282"],
    ["A4 partition", "7.04", "5.89", "1.71×", "3112"],
    ["A6 sort+compact", "19.9", "3.02", "3.33×", "6061"],
    ["A6b coarse+compact", "11.1", "3.15", "3.20×", "5822"],
  ],
  "Table 4.1 — Preparation cost and traversal time, fiction corpus, 18.3 M tokens. † A1 was measured unpinned; its preparation is host-side and unaffected by GPU clock state. Speedup is over the baseline traversal time only; single-pass economics, which charge preparation as well, are treated in Section 4.7."
);

R([
  "Two features of Table 4.1 are immediately contrary to expectation. Exact ordering (A2b) is ",
  { i: "slower" },
  " than the baseline it was supposed to improve, and so is the host sort (A1) that produces the same order. Meanwhile the crudest ordering in the study, a single-pass partition on one leading letter (A4), is the fastest of the ordering-only algorithms. The remainder of this chapter explains that inversion.",
]);

// ---------------------------------------------------------- 4.3
H2("4.3. P1 — saturation of ordering precision");

R([
  { b: "Prediction." },
  " Traversal time is flat in ordering precision beyond the saturation depth of Eq. (14), so coarse orderings capture essentially the whole gain of exact ones.",
], { noIndent: true });

R([
  { b: "Result: confirmed, and in a stronger form than stated." },
  " Ordering precision is not merely subject to diminishing returns; past a very shallow depth it is actively harmful. Arranging the ordering-only algorithms by precision — exact, 8 bytes, 4 bytes, 1 letter — traversal time falls monotonically as precision ",
  { i: "decreases" },
  ": 10.46, 9.93, 8.03, 5.89 ms. The hardware counters show why, and they move in opposite directions.",
]);

R([
  "Both columns of Table 4.2 that measure the benefit of ordering are considerably larger than an earlier version of this work reported, because both had been read from substitute counters. Path sharing is visible directly in requests per token, which the hardware counts ",
  { i: "after" },
  " intra-warp coalescing: exact ordering reduces them from 8.66 to 2.60, so seventy per cent of the memory requests a warp would otherwise issue are eliminated by its lanes descending a common path. Warp efficiency rises from 0.318 to 0.944, a threefold improvement, where branch-target uniformity — the proxy previously used in its place — reports the far milder 72.8 to 98.4 %. The mechanism that ordering exploits is therefore stronger than the earlier figures suggested, which sharpens the result rather than softening it: ordering delivers a threefold improvement in precisely the quantity it targets, and at exact precision is still slower than doing nothing at all.",
]);

T(
  [
    ["Algorithm", "Precision", "Req./token", "Warp eff.", "L1 hit, %", "L2 hit, %", "DRAM read, GB", "Lookup, ms"],
    ["A0 baseline", "none", "8.66", "0.318", "69.0", "95.6", "0.38", "10.07"],
    ["A4 partition", "1 letter", "7.16", "0.377", "62.0", "84.8", "1.34", "5.89"],
    ["A3 prefix", "2 letters", "5.63", "0.473", "56.2", "80.8", "1.43", "8.03"],
    ["A2 sort-8B", "4 letters", "3.50", "0.728", "38.2", "78.0", "1.45", "9.93"],
    ["A2b exact", "exact", "2.60", "0.944", "25.3", "77.6", "1.45", "10.46"],
  ],
  "Table 4.2 — The central measurement, recorded with the ten-counter metric set. Requests per token are counted after intra-warp coalescing, so the third column measures path sharing directly: as precision rises, a warp's lanes increasingly descend a common trie path and their loads collapse into a single request. Warp efficiency is threads retired per executed instruction, divided by 32. Both rise monotonically with precision while cache hit rate falls monotonically. Ordering buys path sharing and pays for it in memory traffic."
);

F(
  "fig4_1_saturation.png", 0.8349,
  "Figure 4.1 — The two opposing trends of Table 4.2, plotted against ordering precision measured in leading Cyrillic letters (two bytes each in UTF-8). Warp uniformity rises monotonically with precision while cache locality falls; traversal time, in the lower panel, is minimised at one letter and returns to the baseline at exact ordering. The dotted line marks baseline traversal time."
);

R([
  "Sorting multiplies DRAM traffic by 3.7. The excess is attributable arithmetically to the token stream rather than to the trie: 18.3 M tokens gathered at roughly two 32-byte sectors each account for about 1.17 GB, against roughly 200 MB when the corpus is streamed in natural order, and the measured excess over baseline is 1.08 GB. This is the ν_S term of Eq. (5) — the coalescing that natural order enjoys for free and that every permutation destroys. Exact ordering wins the largest uniformity gain in the study and still loses overall, because it also pays the largest traffic penalty.",
]);

// ---------------------------------------------------------- 4.4
H2("4.4. P2 — mechanism separation, alphabetical against length");

R([
  { b: "Prediction." },
  " Alphabetical ordering moves the cache counters, length ordering the uniformity counters; and in the strong form, length ordering alone is a net loss because it purchases uniformity by destroying prefix adjacency.",
], { noIndent: true });

R([
  { b: "Result: confirmed in the strong form." },
  " Sweeping all three key modes of Eq. (1)–(2) over the full scale range, the best achievable speedup under each key is:",
]);

T(
  [
    ["Tokens", "alpha, best", "len, best", "lenalpha, best"],
    ["1 M", "6.15× (A6)", "2.40× (A2b)", "5.88× (A6)"],
    ["5 M", "5.89× (A6)", "1.42× (A2b)", "6.52× (A6)"],
    ["20 M", "3.36× (A6)", "0.95× (A2b)", "2.98× (A6)"],
    ["50 M", "3.15× (A6b)", "0.93× (A6b)", "2.26× (A6)"],
  ],
  "Table 4.3 — Best speedup attainable under each key mode. Alphabetical dominates at every scale; beyond 10 M tokens no length-keyed ordering beats the baseline at all."
);

R([
  "The decisive evidence is not this comparison across key modes but a controlled experiment ",
  { i: "within" },
  " the length-keyed mode. Under a length-major key every algorithm loses (0.65–1.16×) with a single exception: A2b reaches 2.50×. This is not an anomaly. A2b's refinement passes sort by the actual token bytes within each length class, so under a length key it silently becomes exact length-then-alphabetical ordering. The key, the code path and the data are identical to the failing variants; the sole difference is whether alphabetical refinement occurs, and the outcome moves from 0.88× to 2.50×. Trip-count uniformity is thus shown to be worth less than nothing on its own — it is worth having only when it does not displace path sharing.",
]);

R([
  "The composite key confirms the reading. Under lenalpha, A6 still reaches 5.88× because it retains seven alphabetical bytes beneath the length byte, but the coarse variants collapse (A6b to 1.55×, A4 to 0.85×), because two bytes of a length-major key encode essentially nothing but the length. Since only the top bytes of the key measurably affect performance (Section 4.3), the alpha/len question reduces to a question of bit budget: what the leading bits of the sort key are spent on. Spending them on the first letter buys path sharing; spending them on length buys uniformity and forfeits path sharing, which is the worse trade at every scale measured.",
]);

// ---------------------------------------------------------- 4.5
H2("4.5. P3 — composition of ordering and compaction");

R([
  { b: "Prediction." },
  " Ordering acts on e and h, compaction on ν_S; being terms of different factors of Eq. (3), their gains should compose multiplicatively.",
], { noIndent: true });

R([
  { b: "Result: confirmed." },
  " The len sweep provides the decomposition directly, and from the same kernel on the same data rather than from a synthetic isolation experiment. Under a length key the ordering supplies no path sharing whatsoever, so the gain of the compacted variant A6b there — 1.16× at 2 M tokens — is pure coalescing. Under an alphabetical key the same algorithm yields 3.64×. The implied path-sharing factor is therefore",
]);

EQ(
  [
    mr("3.64 / 1.16 ≈ 3.14,     so     "),
    sub("G", "total"),
    mr(" ≈ "),
    sub("G", "coalesce"),
    mr(" · "),
    sub("G", "path"),
    mr(" = 1.16 · 3.14."),
  ],
  15
);

R([
  "The counters corroborate the mechanism rather than merely the arithmetic. A6 attains the branch uniformity of full sorting (91.7 % against A2's 91.8 %) while moving ",
  { i: "less" },
  " DRAM traffic than the unsorted baseline (0.38 GB against 0.40 GB), at the lowest sectors-per-request in the study (2.93 against the baseline's 5.63). It therefore holds all three mechanisms of Section 3.1 at once, which no ordering-only algorithm does.",
]);

R([
  "Two counter readings require comment rather than concealment. First, A6 shows the ",
  { i: "lowest" },
  " L2 hit rate of any variant (68.5 %) together with the lowest absolute traffic; hit rate is a ratio, and with sectors-per-request reduced to 2.93 the residual misses form a larger fraction of a far smaller total. Absolute traffic is reported alongside every hit rate for this reason. Second, A6b shows both worse coalescing (4.53 sectors per request) and worse uniformity (78.6 %) than A6, yet performs within 6 % of it and overtakes it beyond 20 M tokens; its compensation is L2 residency (91.2 % against 68.5 %), coarse ordering keeping the trie working set concentrated where fine ordering sweeps the whole vocabulary. The two algorithms reach comparable throughput by different routes, which is why the optimum between them moves with scale.",
]);

// ---------------------------------------------------------- 4.6
H2("4.6. P4 — scale dependence of the optimum");

R([
  { b: "Prediction." },
  " Since only h depends on n, the optimal ordering precision falls as the corpus grows.",
], { noIndent: true });

R([
  { b: "Result: confirmed, with a mechanism opposite to the one assumed." },
  " The prediction was framed on the expectation that the baseline would degrade once the corpus outgrew L2, making ordering progressively more valuable. The baseline does not degrade at all. Its cost per token is 0.56 ns at 1 M tokens and 0.52 ns at 50 M — invariant across a fiftyfold range. What changes is that the ordered variants decay ",
  { i: "toward" },
  " it.",
]);

T(
  [
    ["Tokens", "A0", "A2 sort-8B", "A3 prefix", "A4 partition", "A6 s+c", "A6b coarse+c"],
    ["1 M", "0.56", "0.18", "0.21", "0.27", "0.09", "0.15"],
    ["5 M", "0.54", "0.31", "0.28", "0.30", "0.09", "0.14"],
    ["20 M", "0.53", "0.50", "0.42", "0.35", "0.16", "0.16"],
    ["50 M", "0.52", "0.55", "0.45", "0.36", "0.21", "0.17"],
  ],
  "Table 4.4 — Traversal cost in nanoseconds per token, wiki corpus, each entry the median of five independent process launches (Section 3.6). The baseline is scale-invariant; every ordering degrades, fine orderings fastest. A2 crosses the baseline between 20 M and 35 M tokens and is a net loss thereafter even before preparation is charged."
);

F(
  "fig4_2_scale.png", 0.7765,
  "Figure 4.2 — Per-token traversal cost against corpus size, logarithmic abscissa, wiki corpus, alphabetical key. The baseline is flat across a fiftyfold range; every ordering rises, the finest fastest, and A2 crosses the baseline near 32 M tokens. Only A6b, one letter of precision followed by compaction, holds its cost constant."
);

R([
  "A6b is the only ordering whose per-token cost is also scale-invariant (0.15 → 0.17 ns), and it overtakes A6 at approximately 20 M tokens. The optimum precision falls to one letter by 20 M tokens, and the margin by which it is optimal widens with corpus size — precisely the prediction, reached by the opposite route.",
]);

R([
  "Counters taken at 1 M, 20 M and 50 M tokens identify which factor is responsible. Sectors per request vary by under 3 % across the range for every algorithm (baseline 5.82 → 5.64; A2 4.72 → 4.80; A6 3.10 → 3.08) and branch uniformity by under one point (A2 90.95 → 91.68). The L2 hit fraction is the only counter that moves, and it moves in opposite directions: down about ten points for orderings that permute without re-materializing (A2 89.6 → 79.4; A3 92.0 → 80.7) and ",
  { i: "up" },
  " for the compacted ones (A6b 83.1 → 89.3), while remaining flat in natural order (94.1 → 95.2). The decay is therefore an L2 capacity effect, not a coalescing effect, which is the content of Proposition 2 and the reason the model separates as it does.",
]);

// ---------------------------------------------------------- 4.7
H2("4.7. Single-pass economics and the operating window");

R([
  "Sections 4.3–4.6 charge only traversal. Criterion (8) at k = 1 additionally charges preparation, and it selects a different algorithm. Writing the single-pass ratio as baseline traversal over preparation plus traversal, a value above unity means the ordering pays with no reuse whatever:",
], { noIndent: true });

T(
  [
    ["Tokens", "A2 sort-8B", "A3 prefix", "A4 partition", "A6 s+c", "A6b coarse+c"],
    ["0.5 M", "1.15×", "1.55×", "0.82×", "0.34×", "0.37×"],
    ["1.5 M", "1.69×", "1.94×", "0.88×", "0.55×", "0.59×"],
    ["3 M", "1.41×", "1.75×", "0.88×", "0.56×", "0.65×"],
    ["5 M", "0.85×", "1.20×", "0.81×", "0.51×", "0.66×"],
    ["8 M", "0.71×", "1.01×", "0.78×", "0.48×", "0.64×"],
    ["10 M", "0.68×", "0.95×", "0.77×", "0.47×", "0.66×"],
  ],
  "Table 4.5 — Single-pass speedup, preparation included. Both sorting variants are single-peaked with an interior optimum near 1.5 M tokens; compaction never repays its gather in one pass."
);

F(
  "fig4_3_oneshot.png", 0.6517,
  "Figure 4.3 — Single-pass speedup with preparation charged, fine grid of ten batch sizes. Values above the horizontal line at unity denote a net win with no reuse whatsoever. Both sorting curves are single-peaked; the shaded region is the operating window in which the prefix sort A3 pays for itself on one pass, closing at the interpolated break-even of 8.2 M tokens."
);

R([
  "The prefix sort A3 is a net single-pass win up to approximately 8.2 M tokens, peaking at 1.94× near 1.5–2 M; A2 wins up to 4.1 M. Below the peak, fixed preparation overheads outweigh a small absolute gain; above it, the ordering decays per Section 4.6. Compaction repays its gather only from roughly the second pass onward. Two operating regimes follow, and they exhaust the useful configurations:",
]);

R([
  "— For batches of roughly 1–8 M tokens traversed once, a cheap prefix sort with no compaction is the correct choice, yielding up to 1.94× with no reuse and no memory overhead beyond the permutation array.",
]);

R([
  "— For a resident corpus traversed twice or more, coarse ordering with compaction is correct at any scale, yielding 3.2–3.9× and, uniquely, not degrading as the corpus grows.",
]);

R([
  "Outside those regimes reordering should not be attempted. It is worth stating plainly that this is a negative result for the most common deployment: a single pass over a corpus larger than about 8 M tokens cannot be accelerated by input reordering on this hardware, and a host-fed pipeline is in any case bounded by transfer (approximately 17 ms for 200 MB against 10 ms of traversal at 18.3 M tokens) rather than by traversal.",
]);

// ---------------------------------------------------------- 4.8
H2("4.8. Clock sensitivity as a check on the model");

R([
  "A robustness check across three pinned core frequencies, undertaken to show the ranking invariant to the operating point, yields an independent test of the factorization. Sensitivity below is the fraction of a frequency increase that appears as speedup: unity denotes perfect scaling with core clock, zero denotes complete insensitivity.",
], { noIndent: true });

T(
  [
    ["Algorithm", "Branch unif., %", "1800 MHz", "2400 MHz", "2730 MHz", "Sensitivity"],
    ["A0 baseline", "72.8", "15.20", "11.41", "10.10", "0.98"],
    ["A4 partition", "78.5", "7.27", "6.26", "5.91", "0.45"],
    ["A6b coarse+c", "80.0", "4.00", "3.37", "3.14", "0.52"],
    ["A3 prefix", "83.9", "8.62", "8.18", "8.00", "0.15"],
    ["A6 sort+compact", "91.5", "3.36", "3.10", "3.01", "0.23"],
    ["A2 sort-8B", "91.5", "10.37", "10.05", "9.92", "0.09"],
    ["A2b exact", "98.5", "10.85", "10.25", "10.11", "0.14"],
  ],
  "Table 4.6 — Traversal time, ms, at three pinned core frequencies with memory pinned at 11501 MHz. Rows ordered by warp uniformity. The ranking of algorithms is identical at all three operating points."
);

R([
  "The baseline scales with core frequency at 0.98, essentially one for one, and is therefore issue-bound — as its 72.8 % warp uniformity implies. Every ordered variant scales at 0.52 or below and is therefore memory-bound. Sorting the table by uniformity shows the trend: the three least uniform algorithms carry the three highest sensitivities, the three most uniform the lowest. Divergence implies serialized instruction issue, which is what scales with core clock, so the ",
  { i: "e" },
  " factor of Eq. (3) is confirmed by a frequency knob without reference to any hardware counter. The relation is a trend rather than a fit — A2 and A6 share 91.5 % uniformity yet differ, 0.09 against 0.23 — so divergence is a strong but not exclusive determinant.",
]);

R([
  "One consequence bears on reporting. Because reducing core frequency penalizes only the issue-bound baseline, the headline speedup is operating-point dependent: A6 yields 4.86× at 1800 MHz, 3.66× at 2400 MHz and 3.35× at 2730 MHz. Every speedup quoted in this work is accompanied by its operating point for that reason.",
]);

// ---------------------------------------------------------- 4.9
// ---------------------------------------------------------- 4.9
H2("4.9. Quantitative validation of the cost model");

R([
  "The four predictions above test the model's structure. This section tests its arithmetic. All seven algorithms were profiled at all seven corpus sizes of the scale sweep — forty-nine measurements — and the model fitted by leave-one-scale-out: each fit sees six corpus sizes and predicts the seventh, so every error reported here is on an operating point its fit never saw. Coefficients are estimated on relative rather than absolute error, since traversal cost spans 0.09 to 0.59 ns per token and an unweighted fit would optimize the slowest algorithms while ignoring the fastest, which are the ones the study recommends.",
], { noIndent: true });

R([
  { b: "Equation (3) fails, and the failure is structural rather than numerical." },
  " Its held-out error is 34.5 %, and it identifies the fastest algorithm in five folds of seven. Three separate observations locate the fault in the functional form rather than in the measurements. The error for the baseline is a near-constant offset of 41 to 44 % at every scale rather than a drift, so the model tracks how cost changes but not what it is. The fitted cache latency is negative at several fit points, which is not physically interpretable. And accuracy ",
  { i: "improves" },
  " when the inputs are made worse.",
]);

T(
  [
    ["Model", "Params", "Held-out MAPE", "Worst point", "Fastest picked"],
    ["Eq. (3), three-level L_eff", "3", "34.5 %", "72.5 %", "5 / 7"],
    ["Eq. (3), cruder proxies for ν and e", "3", "23.5 %", "62.5 %", "5 / 7"],
    ["Eq. (3), divergence factor deleted", "2", "28.8 %", "60.4 %", "5 / 7"],
    ["α ν σ", "1", "20.5 %", "53.4 %", "5 / 7"],
    ["Eq. (3′) = α ν σ + β", "2", "13.1 %", "30.3 %", "5 / 7"],
    ["Eq. (3′) divided by e", "2", "23.9 %", "58.8 %", "5 / 7"],
    ["Eq. (3′) with an added DRAM term", "3", "11.6 %", "39.5 %", "5 / 7"],
    ["Null model, mean of the training folds", "0", "66.9 %", "273.6 %", "—"],
  ],
  "Table 4.7 — Leave-one-scale-out validation over forty-nine measurements. A model that becomes more accurate when its factors are replaced by cruder substitutes is not measuring what it claims to measure; rows two and three are that diagnostic, not a recommendation."
);

R([
  "Replacing the hit-rate-weighted latency average of Eq. (6) by the measured exposure of memory latency gives Eq. (3′), at 13.1 % held-out error with one parameter fewer than Eq. (3). The substitution also absorbs the divergence factor: dividing Eq. (3′) by ",
  { i: "e" },
  " makes the fit substantially worse, so warp efficiency is not an independent term of the cost but one of the routes by which a warp comes to expose latency. The cache hit rates cease to earn their place for the same reason, an explicit DRAM term buying only a further 1.5 points.",
]);

R([
  { b: "Separability, tested directly." },
  " Across the fiftyfold range, transactions per token vary by 2.1 % on average over the seven algorithms and by at most 4.3 % for any one of them, and warp efficiency by 1.6 %. The scale dependence of the whole problem is therefore carried by σ alone — and the scale sensitivity of σ reproduces the experimental ranking of Section 4.6 with no reference to any timing.",
]);

T(
  [
    ["Algorithm", "σ at 1 M", "σ at 50 M", "Variation", "Behaviour in Section 4.6"],
    ["A0 baseline", "80.0", "75.5", "1.7 %", "scale-invariant"],
    ["A6b coarse+compact", "23.4", "21.8", "7.8 %", "scale-invariant"],
    ["A4 partition", "45.2", "53.7", "9.1 %", "mild decay"],
    ["A3 prefix-4B", "41.7", "85.4", "27.5 %", "decays"],
    ["A6 sort+compact", "28.1", "54.2", "35.0 %", "decays, overtaken by A6b"],
    ["A2 sort-8B", "50.1", "155.0", "38.7 %", "decays, crosses the baseline"],
    ["A2b exact", "60.1", "223.7", "43.3 %", "worst at every scale"],
  ],
  "Table 4.8 — Latency exposure against corpus size, wiki corpus, with its coefficient of variation across the range. The two algorithms whose per-token cost is measured scale-invariant in Section 4.6 are exactly the two whose exposure is flat, and the ordering of the remaining five by exposure sensitivity is their ordering by observed decay."
);

R([
  { b: "Two limits are conceded." },
  " σ is measured on the kernel it describes, so Eq. (3′) explains and interpolates but does not forecast: the causal chain runs from corpus size to L2 residency to latency exposure to time, and this study measures its endpoints but not its middle link. And the model must not be used to resolve close comparisons — the two compacted algorithms differ by under 20 % beyond 35 M tokens, and which of them a fitted model selects there is not stable across the families of Table 4.7. The A6-to-A6b crossover is an experimental finding, not a prediction of the model.",
]);

H2("4.10. Threats to validity");

R([
  { b: "Single device and single cache capacity." },
  " All measurements come from one GPU with 64 MB of L2. The model of Section 3.3 attributes the scale dependence of the optimum to working set against cache capacity, but only one capacity has been observed; the predicted shift of the crossover with L2 size is untested. This is the most serious limitation and the first that further work should address.",
], { noIndent: true });

R([
  { b: "Single dictionary and single lookup structure." },
  " The trie is fixed at 9.35 M states with the branching structure of Ukrainian morphology, and traversal is byte-wise with a linear transition scan. Whether the findings concern tries specifically or pointer-chasing dictionary lookup generally is not established.",
]);

R([
  { b: "The model interpolates but does not forecast." },
  " Section 4.9 fits Eq. (3′) and validates it against held-out corpus sizes, but σ is obtained from the kernel it describes. Choosing an algorithm for a corpus size that has not been run therefore still requires modelling σ from ordering precision and working-set size, which this study does not attempt.",
]);

R([
  { b: "Unrecognized tokens." },
  " Between 5 % and 16 % of tokens depending on corpus fail to reach a lemma and abort traversal early. All reported figures average over successful and aborted lookups; the two populations have different access patterns and have not been separated.",
]);

R([
  { b: "Virtualized host." },
  " Measurements were taken under WSL2. Device-side timings are unaffected, but host-to-device transfer may be slower than on a bare-metal host, which bears on the transfer-bound argument of Section 4.7 specifically.",
]);

R([
  { b: "An unexplained pipeline regression." },
  " In the streamed configuration A5, enabling the partition makes the pipeline slower, the opposite of the one-shot result. The penalty is not the fixed one an earlier single measurement suggested. Across fourteen runs spanning seven corpus sizes it is present in every one, averages 20.6 %, and grows with scale: between 8 and 17 % up to 35 M tokens, then 52.8 % and 64.7 % at 50 M. The partition work is issued on the copy stream and appears to serialize against the transfer it was meant to overlap, and a penalty that grows with the volume transferred is what that mechanism predicts — which makes the explanation more credible than the fixed offset did, without making it established. The effect is reported rather than resolved, and A5 is accordingly excluded from the recommendations of Section 4.7.",
]);

const doc = new Document({
  creator: "",
  title: "Chapter 4. Results",
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{
    properties: { page: {
      size: { width: 11906, height: 16838 },
      margin: { top: 1134, right: 567, bottom: 1134, left: 1701 },
    } },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("chapter4_results.docx", b);
  console.log("wrote chapter4_results.docx", b.length, "bytes");
});
