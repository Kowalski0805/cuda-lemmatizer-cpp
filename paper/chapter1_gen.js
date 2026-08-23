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


// ============================================================ abstract
H1("Abstract");

R([
  "Lemmatization of large Ukrainian corpora is a dictionary lookup performed hundreds of millions of times, and on a graphics processor it is limited not by arithmetic but by the irregularity of its memory access: neighbouring threads descend unrelated paths of a nine-million-state trie. The obvious remedy is to sort the input so that a warp's tokens share prefixes. This work asks whether that remedy pays, and finds that in its natural form it does not.",
], { noIndent: true });

R([
  "Nine ordering strategies — from an exact host sort through GPU radix sorts of decreasing precision to a single-pass partition, with and without re-materialization of the ordered tokens — are implemented against the production trie of a deployed Ukrainian lemmatizer and compared on three corpora at seven corpus sizes from one to fifty million tokens. Every strategy computes the ordering as a transient index array and scatters results back to original positions, so all nine produce bit-identical output and no sorted copy of the corpus is ever retained.",
]);

R([
  "Three results follow. First, ordering precision is worth nothing beyond a single leading letter, and past roughly twenty-seven million tokens an exact ordering is ",
  { i: "slower" },
  " than no ordering at all, even when the cost of producing it is ignored. Second, the reason is a trade the literature on reordering does not price: any permutation destroys the stream coalescing that natural order enjoys for free, and it purchases path sharing with that loss. Measured separately on the same kernel, coalescing is worth 1.16× and path sharing 3.14×. Third, the two are separable — re-materializing the tokens in permuted order recovers coalescing without surrendering path sharing, and yields 3.33× over the unordered baseline at 6.1 billion tokens per second, with a two-byte key capturing 3.20× of it at 56 % of the preparation cost.",
]);

R([
  "Alphabetical and length-major keys are compared directly, since only the leading bytes of a key measurably affect performance and the two compete for them. Alphabetical ordering dominates at every corpus size, and beyond ten million tokens no length-keyed ordering beats the baseline at all: trip-count uniformity is worth less than nothing when it displaces path sharing. A cost model is proposed, falsified against forty-nine profiled measurements, and corrected; the corrected form predicts held-out operating points to 13 % with two parameters. Two measurement artifacts that each produced a self-consistent but false result are documented, as are two hardware counters whose use as substitutes for the modelled quantities silently corrupts the conclusions drawn from them.",
]);

// ============================================================ 1
H1("1. Introduction");

// ---------------------------------------------------------- 1.1
H2("1.1. Lemmatization as a memory-bound problem");

R([
  "Morphological lemmatization — mapping an inflected word form to its dictionary form — is the first substantive stage of most Ukrainian language-processing pipelines, and for a richly inflected language it is not a small one. A single Ukrainian lemma may surface in dozens of forms, so the dictionary that resolves them is large: the one used throughout this work holds 9 351 209 trie states over a flat lemma buffer of 85.8 MB, 260 MB resident on the device. Applied to a corpus, lemmatization is that dictionary consulted once per token, tens or hundreds of millions of times, with no dependence between consultations.",
], { noIndent: true });

R([
  "That independence is what makes the problem attractive on a graphics processor, and the naive expectation is that it should therefore run at bandwidth. It does not. The traversal descends one trie level per input byte, and the state fetched at each level is determined by the bytes already consumed, so the address of the next access cannot be computed until the current one returns. Neighbouring threads hold unrelated tokens and descend unrelated paths. In natural order the kernel measured here sustains roughly 0.53 nanoseconds per token while moving 23.5 bytes of DRAM traffic for each — a delivered bandwidth two orders of magnitude below what the device can supply. The cost is latency and divergence, not throughput.",
]);

R([
  "This is the familiar shape of pointer-chasing on wide SIMT hardware, and it has an equally familiar proposed remedy.",
]);

// ---------------------------------------------------------- 1.2
H2("1.2. The remedy, and why it is not obviously right");

R([
  "If the tokens assigned to one warp shared their leading bytes, their descents would visit the same trie states, and thirty-two independent chains of dependent loads would collapse into one. Sorting the input alphabetically before traversal makes exactly that arrangement, and it is cheap to do on the device: a radix sort over a key packed from the first eight bytes of each token costs eight linear passes. The same reasoning suggests a second key. A warp executes until its longest token terminates, so grouping tokens of equal length would stop short tokens from idling while long ones finish. Prefix grouping and length grouping are the two candidate orderings, and the question of which is worth more is the axis this work is named for.",
], { noIndent: true });

R([
  "The reasoning is incomplete in a way that turns out to decide the outcome. Reordering does not add structure to an unstructured stream; it ",
  { i: "exchanges" },
  " one kind of structure for another. In natural order, thread ",
  { i: "i" },
  " and thread ",
  { i: "i" },
  "+1 read adjacent bytes of the input buffer, so a warp's token-byte reads are served by a handful of contiguous sectors. That property is not earned — it is what an unpermuted array is — and ",
  { i: "any" },
  " permutation destroys it, because a permuted read gathers from arbitrary offsets. Sorting therefore buys path sharing on the trie stream and pays for it on the token stream, and nothing in the argument for sorting establishes that the exchange is favourable. Measured here, it is not: an exact ordering multiplies DRAM traffic by 3.7 and finishes slower than the baseline it was meant to improve.",
]);

R([
  "Nor is the accounting complete once the two streams are priced, because the ordering must also be produced. A sort is linear in the corpus but so is the traversal it accelerates, and the two constants are within a small factor of each other. Whether ordering pays therefore depends on how many times the ordered data is traversed, and the single-pass case — freshly arrived text, lemmatized once — is a strictly harder test than the reuse case and the more common deployment. Both are treated separately throughout.",
]);

// ---------------------------------------------------------- 1.3
H2("1.3. Approach");

R([
  "Nine ordering algorithms are compared: the unordered baseline; an exact comparison sort on the host; GPU radix sorts at exact, four-letter, two-letter and one-letter precision; a single-pass histogram partition; a streamed pipeline in which ordering overlaps transfer; and two algorithms that sort and then re-materialize the tokens in the sorted order. Each is run against the production trie rather than a model of one, reproducing the deployed traversal kernel byte for byte, including its linear scan over unsorted transition lists — the paper measures the system that exists, not an idealized one.",
], { noIndent: true });

R([
  "One design decision is shared by all nine and is what makes them comparable. The ordering exists only as a transient index array: the kernel reads token π(",
  { i: "i" },
  ") and scatters its result to output position π(",
  { i: "i" },
  "), so the output is always in original stream order and no reordered copy of the corpus is retained. Every variant therefore computes the same function, verified by a single checksum over the result vector, and the differences between them are confined to the cost of computing π and the memory behaviour it induces. The rebuild cost that makes maintaining a sorted corpus impractical never arises.",
]);

R([
  "Measurements are taken on an RTX 4080 SUPER with 64 MB of L2, on three Ukrainian corpora — news articles, fiction and an encyclopaedic dump — spanning a factor of 5.6 in type/token ratio, at seven corpus sizes from one to fifty million tokens, with hardware counters collected for every algorithm at every size. Two features of the protocol proved indispensable and are reported as results in their own right, because omitting either produced a self-consistent but entirely false conclusion.",
]);

// ---------------------------------------------------------- 1.4
H2("1.4. Contributions");

R([
  { b: "A negative result on ordering precision." },
  " The value of ordering saturates at a single leading letter. An exact ordering, produced on the device at full depth, is 0.96× the baseline on 18.3 million tokens; a four-letter sort is 1.01×; a one-letter partition is 1.71×. Beyond roughly twenty-seven million tokens the exact ordering is a net loss even with its preparation cost set to zero. Precision is not merely subject to diminishing returns — it is harmful, and increasingly so with corpus size.",
], { noIndent: true });

R([
  { b: "The mechanism, measured rather than inferred." },
  " Ordering exchanges stream coalescing for path sharing. Both sides of the exchange are quantified on the same kernel and the same data: under a length-major key, which provides no prefix grouping whatsoever, compaction alone yields 1.16×, and the same algorithm under an alphabetical key yields 3.64×, so path sharing accounts for the residual 3.14× and the two effects compose multiplicatively. Requests per token, counted after intra-warp coalescing, fall from 8.66 to 2.60 across the precision ladder, and warp execution efficiency rises from 0.318 to 0.944 — a threefold improvement in the quantity ordering targets, obtained by an algorithm that is nonetheless slower than doing nothing.",
]);

R([
  { b: "An algorithm that obtains both." },
  " Because the two effects are separable, re-materializing the tokens in permuted order recovers coalescing without giving back path sharing. Sorting and compacting yields 3.33× over the baseline, 6.1 billion tokens per second on the production trie, at less DRAM traffic than the unordered baseline itself; and a two-byte key captures 3.20× of that at 56 % of the preparation cost. Coarse ordering with compaction is moreover the only strategy measured to be scale-invariant, holding 0.15 to 0.16 nanoseconds per token across a fiftyfold range in corpus size while every other ordering decays.",
]);

R([
  { b: "An answer to the alphabetical-against-length question." },
  " Since only the leading bytes of a key measurably affect performance, the two candidate orderings compete for the same bits. Alphabetical ordering dominates at every corpus size tested, and beyond ten million tokens no length-keyed ordering beats the baseline at all. The comparison is also made as a controlled experiment within a single key mode, where the sole difference between a 0.88× result and a 2.50× result is whether alphabetic refinement is applied inside length classes. The question is therefore not which grouping helps, but what the leading bits of the key are spent on; the measurement says path sharing, decisively, and says trip-count uniformity is worth less than nothing because it displaces path sharing.",
]);

R([
  { b: "An operating window for deployment." },
  " Charging preparation in full, a cheap prefix sort is a net single-pass win below approximately 8.2 million tokens per batch, peaking at 1.94× near 1.5 million; compaction repays its gather only from the second traversal onward. Two regimes exhaust the useful configurations, and which applies is decided by the batch size a deployment actually sees.",
]);

R([
  { b: "A cost model, falsified and corrected." },
  " A three-factor decomposition of per-token cost is proposed, then tested by leave-one-scale-out over forty-nine profiled measurements. It fails at 34.5 % held-out error, and the failure is shown to be structural rather than numerical — the model becomes ",
  { i: "more" },
  " accurate when its factors are replaced by cruder substitutes. Replacing its hit-rate-weighted latency average by a direct measurement of latency exposure gives a two-parameter law that predicts held-out corpus sizes to 13.1 %, and subsumes the divergence term rather than competing with it. The separability the model rests on is confirmed more strongly than it was stated: transactions per token vary by 2.1 % across a fiftyfold range, so the entire scale dependence of the problem is carried by one scalar per algorithm.",
]);

R([
  { b: "Two measurement artifacts and two counter substitutions." },
  " On a consumer device the clocks cannot be pinned without administrative privileges, and an idle GPU drops to 210 MHz against a maximum of 3105; any host-side work between timed kernels parks them, and timing the variants in declaration order produced a clean monotone ranking that tracked ",
  { i: "execution order" },
  " and nothing else. Separately, preparation timed once absorbs per-process initialization costs, and correcting it moved this work's headline break-even estimate by 28 %. Two hardware counters are also shown to be unusable as substitutes for the quantities they resemble: branch-target uniformity understates warp efficiency threefold, and sectors per request omits how many requests a token issues, a quantity differing fourfold between the algorithms compared. A cost model fitted on the latter returns negative memory latencies.",
]);

// ---------------------------------------------------------- 1.5
H2("1.5. Organization");

R([
  "Chapter 2 positions the work against the literature on locality-aware reordering, GPU sorting primitives and divergence mitigation, and against existing Ukrainian morphological analysis. Chapter 3 states the problem formally, develops the cost model and its correction, defines the nine algorithms, derives four falsifiable predictions, and specifies the measurement protocol. Chapter 4 reports the experiments, organized as evidence for or against each prediction, and validates the model quantitatively. Chapter 5 concludes and states what the results imply for the deployed system.",
], { noIndent: true });


const doc = new Document({
  creator: "",
  title: "Abstract and Chapter 1. Introduction",
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
  fs.writeFileSync("chapter1_introduction.docx", b);
  console.log("wrote chapter1_introduction.docx", b.length, "bytes");
});
