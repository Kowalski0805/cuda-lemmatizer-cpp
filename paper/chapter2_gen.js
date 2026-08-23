// chapter2_gen.js — generated chapter. Run: node chapter2_gen.js
// NOTE: bibliographic details below were written from memory and were NOT
// verified against the sources. Check every author list, venue and year
// before submission.
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

// ============================================================ 2
H1("2. Related work");

R([
  "The problem addressed here sits at the intersection of four literatures that rarely cite one another: locality-aware reordering, which establishes that permuting data to improve cache behaviour is sometimes profitable and sometimes not; GPU sorting, which supplies the mechanism and sets the price; work on irregularity and divergence in SIMT execution, which explains why the price might be worth paying; and morphological analysis for Slavic languages, which supplies the application. This chapter reviews each and states precisely what the present work adds.",
], { noIndent: true });

// ---------------------------------------------------------- 2.1
H2("2.1. Locality-aware reordering");

R([
  "The idea that a permutation can buy cache locality is long established, and graph analytics is where it has been studied most systematically. Gorder (Wei et al., 2016) constructs a vertex ordering that maximizes the number of shared neighbours between vertices placed close together, reporting substantial speedups across graph kernels at the cost of an expensive preprocessing step. Rabbit Order (Arai et al., 2016) pursues the same goal at far lower cost through hierarchical community detection, making the ordering cheap enough to compute just in time. Cagra (Zhang et al., 2017) reorders and segments the vertex space so that the working set of each segment fits in the last-level cache.",
], { noIndent: true });

R([
  "The most directly relevant contribution to the present work is a negative one. Balaji and Lucia (2018) evaluate lightweight reorderings across a range of graphs and find that whether reordering pays depends on properties of the input that the reordering itself does not control, and that for many inputs the preprocessing is never amortized. Faldu et al. (2019) reach a similar conclusion and attribute the variance to how much of the working set was already cache-resident. The pattern this work reports — a reordering that is theoretically well motivated, measurably achieves the property it targets, and still loses — is the same pattern, arrived at independently in a different domain.",
]);

R([
  "Two differences of substance separate this work from that literature. First, graph reordering permutes the ",
  { i: "data structure" },
  ": vertex identifiers are renumbered and the adjacency representation rebuilt, a cost paid once and amortized over all subsequent queries. Here the data structure — the trie — is fixed and shared, and what is permuted is the ",
  { i: "query stream" },
  ". The amortization argument is therefore inverted: there is no persistent artifact to reuse unless the same corpus is traversed again, which makes the single-pass case the primary one rather than a corner case. Second, and more consequentially, the graph reordering literature is written for CPUs, where the cost this work identifies as decisive does not exist. A CPU core reading a permuted index array loses spatial locality but has nothing analogous to warp-level coalescing to forfeit. On SIMT hardware, an unpermuted array is served to a warp in a few contiguous sectors, and that property is destroyed by any permutation whatsoever. This cost is not priced anywhere in the reordering literature because on the hardware that literature targets there is nothing to price.",
]);

R([
  "Outside graph analytics, the same idea appears as standard practice without being framed as reordering research. Particle simulations sort particles into spatial cells each timestep so that neighbouring threads read neighbouring memory, at a cost accepted as routine. In database systems, Zhou and Ross (2003; 2004) buffer accesses to memory-resident index structures so that operations reaching the same node are performed together — a reordering of the access stream rather than of the structure, and thus the closest classical antecedent to what is done here, though performed to fill cache lines on a single core rather than to fill a warp.",
]);

// ---------------------------------------------------------- 2.2
H2("2.2. Sorting on GPUs");

R([
  "The cost side of the trade-off rests on the state of GPU sorting. Satish et al. (2009) established radix sort as the method of choice for fixed-width keys on manycore hardware, and Merrill and Grimshaw (2011) brought it to close to memory-bandwidth efficiency with the design that underlies the CUB library used in this work. A least-significant-digit radix sort with ",
  { i: "r" },
  "-bit digits costs a number of linear passes proportional to the key width, which is the property this work exploits: restricting the sorted bit range is a direct and continuous dial between ordering precision and preparation cost, and it is what makes the saturation question empirically answerable rather than a choice between sorting and not sorting.",
], { noIndent: true });

R([
  "Sorting as an enabling step for locality is familiar in GPU database work, where radix partitioning precedes joins and aggregations so that each partition's working set fits in cache. The present work borrows the partitioning step directly, but differs in what is being made local: a join partitions so that build and probe sides meet, whereas here the partition exists solely to make a warp's threads descend a shared path, and the payload — the trie — is never partitioned at all.",
]);

// ---------------------------------------------------------- 2.3
H2("2.3. Irregularity and divergence in SIMT execution");

R([
  "Why grouping similar work together should help at all is the subject of a substantial architecture literature. Fung et al. (2007) introduced dynamic warp formation, regrouping threads at run time so that those following the same control-flow path execute together; Meng et al. (2010) proposed dynamic warp subdivision to tolerate both branch and memory divergence. Han and Abdelrahman (2011) treat the problem in software through code transformations that reduce divergent branches. Burtscher et al. (2012) quantify irregularity across a suite of GPU programs and separate control-flow irregularity from memory irregularity, showing the two need not co-occur — a separation this work confirms in a specific setting and, in the case of exact ordering, finds to be actively opposed.",
], { noIndent: true });

R([
  "The closest antecedent is G-Streamline (Zhang et al., 2011), which eliminates dynamic irregularities on GPUs by reordering data and threads on the fly, with transformations chosen to keep the reordering cost below the benefit. The present work can be read as a careful negative case study for that programme: the reordering is applied, it demonstrably achieves the regularity it targets — requests per token fall from 8.66 to 2.60, warp execution efficiency rises from 0.318 to 0.944 — and the transformation is nonetheless a net loss, because the reordering itself introduces a second irregularity in the input stream that the first does not compensate. That the remedy can create the disease it treats appears not to have been reported quantitatively before.",
]);

R([
  "Compaction, the step that resolves the trade in this work, is itself a known technique in a different guise: stream compaction is a standard GPU primitive, and gathering scattered data into contiguous buffers before a kernel is routine advice for coalescing. What appears to be new is the observation that it must be combined with reordering rather than chosen instead of it, and that the two act on separable factors whose gains compose multiplicatively.",
]);

// ---------------------------------------------------------- 2.4
H2("2.4. Dictionary structures and Ukrainian morphological analysis");

R([
  "The lookup structure follows a long line of work on finite-state dictionaries. Daciuk et al. (2000) gave the incremental construction of minimal acyclic finite-state automata that underlies most modern morphological dictionaries, including the directed acyclic word graph used as this system's CPU-side reference. The GPU-resident structure here is a flat trie rather than a minimized automaton, trading space for a traversal whose memory pattern is uniform enough to vectorize — a trade that is only defensible when the dictionary fits comfortably in device memory, as it does at 260 MB.",
], { noIndent: true });

R([
  "For Ukrainian specifically, morphological analysis is dominated by dictionary-driven systems. The VESUM dictionary maintained by Rysin and Starko is the principal open lexical resource and underlies the Ukrainian modules of LanguageTool; pymorphy2 (Korobov, 2015) provides analysis and generation for Russian and Ukrainian from a compressed automaton. Neural pipelines — UDPipe (Straka and Straková, 2017) and Stanza (Qi et al., 2020) — perform lemmatization as a sequence-labelling or seq2seq task and reach competitive accuracy, but at a per-token cost orders of magnitude above dictionary lookup, which is precisely why the dictionary path remains the one worth optimizing for corpus-scale batch processing. None of this literature treats the throughput of the lookup itself as the object of study; accuracy is the reported metric throughout.",
]);

// ---------------------------------------------------------- 2.5
H2("2.5. Position of this work");

R([
  "Against that background the contribution of this work can be stated precisely. The reordering literature establishes that permutations buy locality and sometimes fail to repay their cost; it does not price the loss of warp-level coalescing, because it targets hardware where that loss does not occur. The divergence literature establishes that regularizing work helps and proposes reordering as a means; it does not report a case in which the reordering achieves its stated aim and loses anyway. The GPU sorting literature supplies the mechanism and its cost but is silent on what precision is worth buying. And the Ukrainian morphology literature supplies the application without treating lookup throughput as a research object at all.",
], { noIndent: true });

R([
  "This work measures the exchange rather than assuming its direction: it prices both sides on the same kernel and the same data, finds the ordering precision at which the benefit saturates to be a single letter, identifies the corpus size beyond which exact ordering becomes a net loss, gives an algorithm that captures both effects at once, and reports the batch-size window within which the cheap version repays itself on a single pass. It also reports, as findings rather than as apparatus, two measurement artifacts and two hardware-counter substitutions each of which produced a self-consistent but false conclusion — a class of result the performance literature discusses less often than it encounters.",
]);

// ---------------------------------------------------------- refs
H2("References cited in this chapter");

REF("Arai, J., Shiokawa, H., Yamamuro, T., Onizuka, M., Kitsuregawa, M. Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis. IPDPS, 2016.");
REF("Balaji, V., Lucia, B. When is Graph Reordering an Optimization? Studying the Effect of Lightweight Graph Reordering on Input Graph Structure and Cache Locality. IISWC, 2018.");
REF("Burtscher, M., Nasre, R., Pingali, K. A Quantitative Study of Irregular Programs on GPUs. IISWC, 2012.");
REF("Daciuk, J., Mihov, S., Watson, B. W., Watson, R. E. Incremental Construction of Minimal Acyclic Finite-State Automata. Computational Linguistics, 26(1), 2000.");
REF("Faldu, P., Diamond, J., Grot, B. A Closer Look at Lightweight Graph Reordering. IISWC, 2019.");
REF("Fung, W. W. L., Sham, I., Yuan, G., Aamodt, T. M. Dynamic Warp Formation and Scheduling for Efficient GPU Control Flow. MICRO, 2007.");
REF("Han, T. D., Abdelrahman, T. S. Reducing Branch Divergence in GPU Programs. GPGPU-4, 2011.");
REF("Korobov, M. Morphological Analyzer and Generator for Russian and Ukrainian Languages. AIST, 2015.");
REF("Meng, J., Tarjan, D., Skadron, K. Dynamic Warp Subdivision for Integrated Branch and Memory Divergence Tolerance. ISCA, 2010.");
REF("Merrill, D., Grimshaw, A. High Performance and Scalable Radix Sorting. Parallel Processing Letters, 21(2), 2011.");
REF("Qi, P., Zhang, Y., Zhang, Y., Bolton, J., Manning, C. D. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. ACL System Demonstrations, 2020.");
REF("Satish, N., Harris, M., Garland, M. Designing Efficient Sorting Algorithms for Manycore GPUs. IPDPS, 2009.");
REF("Straka, M., Straková, J. Tokenizing, POS Tagging, Lemmatizing and Parsing UD 2.0 with UDPipe. CoNLL Shared Task, 2017.");
REF("Wei, H., Yu, J. X., Lu, C., Lin, X. Speedup Graph Processing by Graph Ordering. SIGMOD, 2016.");
REF("Zhang, Y., Kiriansky, V., Mendis, C., Amarasinghe, S., Zaharia, M. Making Caches Work for Graph Analytics. IEEE International Conference on Big Data, 2017.");
REF("Zhang, E. Z., Jiang, Y., Guo, Z., Tian, K., Shen, X. On-the-Fly Elimination of Dynamic Irregularities for GPU Computing. ASPLOS, 2011.");
REF("Zhou, J., Ross, K. A. Buffering Accesses to Memory-Resident Index Structures. VLDB, 2003.");

const doc = new Document({
  creator: "",
  title: "Chapter 2. Related work",
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
  fs.writeFileSync("chapter2_related_work.docx", b);
  console.log("wrote chapter2_related_work.docx", b.length, "bytes");
});
