// Gera docs/treino_cased.docx a partir do .md: titulo, secoes, paragrafos,
// listas, tabelas, `codigo` e [links]. So o subconjunto que o doc usa.
// Uso: npm install docx && node scripts/md_para_docx.js entrada.md saida.docx
const fs = require("fs");
const d = require("docx");

const [, , entrada, saida] = process.argv;
if (!entrada || !saida) {
  throw new Error(`uso: node md_para_docx.js entrada.md saida.docx (recebido: ${entrada}, ${saida})`);
}
const linhas = fs.readFileSync(entrada, "utf8").split("\n");
const LARGURA = 9026; // A4 com margens de 1440 DXA

function runs(texto, extra = {}) {
  const out = [];
  const re = /(`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let ult = 0;
  let m;
  while ((m = re.exec(texto))) {
    if (m.index > ult) out.push(new d.TextRun({ text: texto.slice(ult, m.index), ...extra }));
    const t = m[0];
    if (t.startsWith("`")) {
      out.push(new d.TextRun({ text: t.slice(1, -1), font: "Consolas", ...extra }));
    } else {
      const [, rotulo, url] = t.match(/\[([^\]]+)\]\(([^)]+)\)/);
      out.push(new d.ExternalHyperlink({ link: url, children: [new d.TextRun({ text: rotulo, style: "Hyperlink", ...extra })] }));
    }
    ult = m.index + t.length;
  }
  if (ult < texto.length) out.push(new d.TextRun({ text: texto.slice(ult), ...extra }));
  return out;
}

function tabela(bloco) {
  const celulas = bloco
    .filter((_, i) => i !== 1)
    .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const n = celulas[0].length;
  const larg = Math.floor(LARGURA / n);
  const borda = { style: d.BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
  return new d.Table({
    width: { size: larg * n, type: d.WidthType.DXA },
    columnWidths: Array(n).fill(larg),
    rows: celulas.map((linha, i) => new d.TableRow({
      tableHeader: i === 0,
      children: linha.map((c) => new d.TableCell({
        width: { size: larg, type: d.WidthType.DXA },
        borders: { top: borda, bottom: borda, left: borda, right: borda },
        shading: i === 0 ? { type: d.ShadingType.CLEAR, fill: "E8EEF4" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new d.Paragraph({ children: runs(c, { size: 18, bold: i === 0 }) })],
      })),
    })),
  });
}

const corpo = [];
for (let i = 0; i < linhas.length; i++) {
  const l = linhas[i];
  if (!l.trim()) continue;
  if (l.startsWith("# ")) corpo.push(new d.Paragraph({ heading: d.HeadingLevel.TITLE, children: runs(l.slice(2)) }));
  else if (l.startsWith("## ")) corpo.push(new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, children: runs(l.slice(3)) }));
  else if (l.startsWith("- ")) corpo.push(new d.Paragraph({ numbering: { reference: "bol", level: 0 }, children: runs(l.slice(2)) }));
  else if (l.startsWith("|")) {
    const bloco = [];
    while (i < linhas.length && linhas[i].startsWith("|")) bloco.push(linhas[i++]);
    i--;
    corpo.push(tabela(bloco));
    corpo.push(new d.Paragraph({ children: [] }));
  } else corpo.push(new d.Paragraph({ spacing: { after: 120 }, children: runs(l) }));
}

const doc = new d.Document({
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } } },
  numbering: { config: [{ reference: "bol", levels: [{ level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: corpo }],
});
d.Packer.toBuffer(doc).then((b) => fs.writeFileSync(saida, b));
