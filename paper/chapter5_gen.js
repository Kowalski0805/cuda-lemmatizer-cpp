// chapter5_gen.js — generated chapter. Run: node chapter5_gen.js
// chapter1_gen.js — abstract and Chapter 1 (Introduction).
// Run: node chapter1_gen.js  ->  chapter1_introduction.docx
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


// Bibliography entry: hanging indent, no first-line indent.
const refPara = (text) =>
  new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: 240, after: 60 },
    indent: { left: 567, hanging: 567 },
    children: [new TextRun({ text, font: FONT, size: 24 })],
  });
const REF = (t) => body.push(refPara(t));

// ============================================================ 5
H1("5. Conclusions");

R([
  "This work asked whether sorting the input stream of a GPU trie lemmatizer pays for itself, and if so, on which key. The answer has three parts, and none of them is the expected one.",
], { noIndent: true });

// ---------------------------------------------------------- 5.1
H2("5.1. Findings");

R([
  { b: "Ordering precision is worth nothing beyond one letter, and eventually less than nothing." },
  " Across the ladder from exact ordering down to a single-pass partition on one leading letter, traversal time falls as precision ",
  { i: "decreases" },
  ": 10.46, 9.93, 8.03 and 5.89 ms on 18.3 million tokens. An exact ordering produced entirely on the device is slower than no ordering at all, and beyond approximately 27 million tokens it remains a net loss even when its preparation is charged nothing. The cost of ordering is not merely unrecovered at high precision; the ordering itself becomes harmful.",
], { noIndent: true });

R([
  { b: "The reason is an exchange, not an improvement." },
  " Reordering does not add structure to the stream — it trades stream coalescing, which an unpermuted array possesses for free and which any permutation destroys, for path sharing on the trie. Both sides were priced on the same kernel and the same data: coalescing is worth 1.16× and path sharing 3.14×, and they compose multiplicatively. Exact ordering wins the largest path-sharing gain in the study, reducing requests per token from 8.66 to 2.60 and raising warp execution efficiency from 0.318 to 0.944, and still loses, because it also pays the largest coalescing penalty — a 3.7-fold increase in DRAM traffic.",
]);

R([
  { b: "Both can be had at once." },
  " Because the two effects act on separable factors, re-materializing the tokens in the permuted order recovers coalescing without surrendering path sharing. Sorting with compaction yields 3.33× over the baseline at 6.1 billion tokens per second, moving ",
  { i: "less" },
  " DRAM traffic than the unordered baseline; a two-byte key captures 3.20× of that at 56 % of the preparation cost, and is the only strategy measured to be scale-invariant, holding 0.15 to 0.16 nanoseconds per token across a fiftyfold range in corpus size.",
]);

R([
  { b: "On the alphabetical-against-length question, the answer is decisive." },
  " Because only the leading bytes of a key measurably affect performance, the two orderings compete for the same bits, and the question is not which grouping helps but what those bits are spent on. Alphabetical ordering dominates at every corpus size tested, and beyond ten million tokens no length-keyed ordering beats the baseline at all. A controlled experiment within a single key mode isolates the mechanism: the sole difference between a 0.88× result and a 2.50× result is whether alphabetic refinement is applied within length classes. Trip-count uniformity is worth less than nothing when purchasing it displaces path sharing.",
]);

R([
  { b: "The effect is governed by corpus size, not corpus composition." },
  " Three corpora spanning a factor of 5.6 in type/token ratio and seventeen points of dictionary recognition rate agree on baseline cost to within 0.04 nanoseconds per token at matched token counts and rank every algorithm identically. What appeared in earlier measurements to be a strong genre dependence proved on inspection to be a scale dependence observed at two different corpus sizes.",
]);

// ---------------------------------------------------------- 5.2
H2("5.2. What the cost model does and does not deliver");

R([
  "The three-factor decomposition of Section 3.3 was tested rather than assumed, by leave-one-scale-out over forty-nine profiled measurements. Its separability claim survives in a stronger form than it was stated: transactions per token vary by 2.1 % across a fiftyfold range of corpus size and warp efficiency by 1.6 %, so the whole scale dependence of the problem is carried by a single scalar per algorithm. Its arithmetic does not survive. Equation (3) predicts held-out operating points to only 34.5 %, and the failure is structural rather than numerical: the model becomes more accurate when its factors are replaced by cruder substitutes, which is the signature of a wrong functional form. Replacing its hit-rate-weighted latency average with a direct measurement of latency exposure gives a two-parameter law accurate to 13.1 % that subsumes the divergence term rather than competing with it.",
], { noIndent: true });

R([
  "Two limits were identified and are not resolved here. The exposure term is measured on the kernel it describes, so the corrected model explains and interpolates but does not forecast; a predictive form requires modelling exposure from ordering precision and working-set size, and this study measures the endpoints of that causal chain but not its middle link. And the model must not be used to resolve close comparisons: the two compacted algorithms differ by less than 20 % beyond 35 million tokens, and which of them a fitted model prefers there is not stable across model families.",
]);

// ---------------------------------------------------------- 5.3
H2("5.3. Consequences for the deployed system");

R([
  "The results select an algorithm as a function of one deployment parameter — the number of tokens in a batch — and the two regimes exhaust the useful configurations. For batches below roughly 8.2 million tokens traversed once, a prefix sort with no compaction is a net win with no reuse whatsoever, peaking at 1.94× near 1.5 million; above that size, or whenever the same corpus is traversed more than once, coarse ordering with compaction gives 3.2 to 3.6× and does not decay with scale. No other configuration is ever the right answer.",
], { noIndent: true });

R([
  "Both map onto operations the production pipeline already performs. The lemmatizer receives sentence columns from its Java caller, splits and explodes them into a token column, and traverses. Inserting a key kernel and a radix sort between the explode and the traversal implements the first regime; adding a gather of the token column, an operation the dataframe library already provides, implements the second. The gather allocates a second copy of the token bytes and is therefore the only configuration with a memory cost worth stating.",
]);

R([
  { b: "The pipeline was instrumented to find out, and the answer qualifies the recommendation sharply." },
  " Recording per-stage timings across batches of 50 000 to 4 million tokens, with the one-time trie initialization excluded, gives the following division of pipeline time.",
]);

T(
  [
    ["Tokens per call", "Split", "Explode", "Traversal", "Group", "Join", "Total, ms", "Traversal share"],
    ["50 000", "1.51", "0.18", "1.43", "0.66", "0.52", "4.30", "33.4 %"],
    ["200 000", "3.23", "1.32", "2.25", "3.17", "1.05", "11.01", "20.4 %"],
    ["1 000 000", "5.63", "2.83", "3.81", "6.08", "1.53", "19.89", "19.2 %"],
    ["4 000 000", "13.75", "5.54", "10.25", "12.08", "4.87", "46.49", "22.0 %"],
  ],
  "Table 5.1 — Stage times in milliseconds for the deployed sentence pipeline, mean over three calls at each size after the initializing call. Stages are synchronized to attribute time correctly, so totals are pessimistic relative to a pipelined execution; the proportions are the quantity of interest."
);

R([
  "The trie traversal that this entire study optimizes accounts for 21.5 % of pipeline time. The remainder is dataframe string handling: splitting sentences into word lists takes 29.4 %, regrouping lemmas by sentence 27.1 %, exploding 12.2 %, and rejoining 9.7 %. By Amdahl's law the 3.33× traversal speedup of Table 4.1 is worth 1.18× end to end, the single-pass 1.94× is worth 1.12×, and a traversal made ",
  { i: "free" },
  " would be worth 1.27×. This does not diminish the traversal results, which concern the traversal, but it does relocate the engineering priority: for this pipeline as it stands, the split and groupby stages are each individually a larger target than the kernel, and any deployment decision about ordering should be taken with that ceiling in view. Fusing the split with the traversal, so that tokens are never materialized as a separate column, would remove two of the four surrounding stages and is the change most likely to matter next.",
]);

R([
  { b: "The fusion was implemented, and it realizes nearly the whole of that ceiling." },
  " If a token is a span of the input sentence and a lemma is a span of the trie's lemma buffer, then nothing between them needs to exist as a dataframe column, and all four surrounding stages can be removed at once rather than merely the first. The fused path counts fields per sentence, scans to obtain token offsets, writes each token as a pointer and a length, traverses the trie exactly as the deployed kernel does, scans the resulting lemma lengths to obtain output positions, and copies. Only one of its five kernels touches the trie; the others are byte scans and copies, and the two large scans are bandwidth-bound.",
]);

T(
  [
    ["Tokens per call", "Staged, ms", "Fused, ms", "Speedup", "Output identical"],
    ["200 000", "12.49", "1.70", "7.35×", "yes"],
    ["1 000 000", "22.81", "4.75", "4.80×", "yes"],
    ["4 000 000", "46.04", "10.69", "4.31×", "yes"],
    ["10 000 000", "99.30", "23.37", "4.25×", "yes"],
  ],
  "Table 5.2 — Fused against staged sentence-to-sentence lemmatization, median of three calls after the initializing call. The two columns are compared bit for bit on the device at every measurement; splitting semantics were separately verified against split_record and join_list_elements on empty strings and on leading, trailing and consecutive separators."
);

R([
  "At four million tokens the staged pipeline needs 46.04 ms, of which 10.25 ms is traversal; the fused path completes the entire operation in 10.69 ms — that is, in about the time the traversal alone previously took. The ratio of total to traversal time in Table 5.1 puts the ceiling on removing the surrounding stages at 4.53× for that batch size, and the measured 4.31× realizes 95 % of it, the shortfall being the fused path's own scans and copies. The end-to-end gain is therefore an order of magnitude larger than anything available from reordering the traversal, which is the practical conclusion of this chapter: for the pipeline as deployed, the ordering question, on which this work spent its effort, was worth at most 1.18×, while deleting four dataframe operations was worth 4.3×.",
]);

R([
  "The two are not alternatives. Fusion removes the plumbing and leaves the traversal, which then constitutes almost the whole of the remaining cost — so the ordering results of Chapter 4, which were bounded to 1.18× against the staged pipeline, apply to nearly the full runtime of the fused one. The correct order of work is fusion first, ordering second, and the second is worth substantially more after the first than before it.",
]);

R([
  "Three caveats attach to Tables 5.1 and 5.2. Sentences were assembled at a fixed twelve words each, and the split and group stages both scale with sentence count rather than token count, so a corpus of different sentence length would shift the proportions and with them the fusion gain. The stage timings of Table 5.1 synchronize between stages and therefore remove whatever overlap the pipelined execution achieves, though Table 5.2 compares two complete pipelines and is free of that objection. And the distribution of batch sizes reported here is the one the probe was asked to submit, not the one the production caller produces — obtaining that requires running the Java application against the instrumented library, which writes the distribution and its regime recommendation without further intervention.",
]);

// ---------------------------------------------------------- 5.4
H2("5.4. Limitations and further work");

R([
  "One device and one cache capacity were measured. The mechanism attributes the scale dependence of the optimum to working set against last-level cache capacity, so the crossover should move with that capacity — a prediction that a second device would test directly and that is the most valuable single experiment remaining. One dictionary and one traversal structure were used, leaving open whether the findings concern tries specifically or pointer-chasing dictionary lookup in general. Between 5 % and 16 % of tokens fail to reach a lemma and abort their traversal early, and successful and aborted lookups, which have different access patterns, were not separated. The streamed pipeline exhibits a regression when partitioning is enabled that is explained but not repaired, and is excluded from the recommendations accordingly.",
], { noIndent: true });

R([
  "Beyond these, two directions follow naturally. Modelling latency exposure from ordering precision and working-set size would close the gap between a model that explains and one that forecasts, and would make the choice of algorithm computable rather than measurable. And the exchange identified here — that any permutation of a query stream forfeits warp-level coalescing, and that the forfeit can be bought back by compaction — is not specific to lemmatization or to tries. It should hold wherever a SIMT kernel is fed an irregular stream of independent queries against a shared structure, which is a considerably larger class of problems than the one measured.",
]);

const doc = new Document({
  creator: "",
  title: "Chapter 5. Conclusions",
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
  fs.writeFileSync("chapter5_conclusion.docx", b);
  console.log("wrote chapter5_conclusion.docx", b.length, "bytes");
});
