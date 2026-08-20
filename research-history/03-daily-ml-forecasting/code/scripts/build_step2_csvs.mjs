import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const intermediate = path.join(root, "data", "processed", "step2_sector_universe");

const specs = [
  { json: "etf_metadata.json", output: ["data", "metadata", "etf_metadata.csv"], sheet: "ETF Metadata" },
  { json: "clean_sector_etf_candidates.json", output: ["data", "metadata", "clean_sector_etf_candidates.csv"], sheet: "Clean Candidates" },
  { json: "monthly_eligible_universe.json", output: ["results", "monthly_eligible_universe.csv"], sheet: "Monthly Eligible" },
  { json: "monthly_sector_representatives.json", output: ["results", "monthly_sector_representatives.csv"], sheet: "Representatives" },
  { json: "classification_review.json", output: ["results", "classification_review.csv"], sheet: "Classification Review" },
  { json: "rejected_sector_candidates.json", output: ["results", "rejected_sector_candidates.csv"], sheet: "Rejected Candidates" },
  { json: "monthly_sector_candidate_coverage.json", output: ["results", "monthly_sector_candidate_coverage.csv"], sheet: "Sector Coverage" },
  { json: "step2_automated_checks.json", output: ["results", "step2_automated_checks.csv"], sheet: "Automated Checks" },
];

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const rendered = typeof value === "boolean" ? (value ? "True" : "False") : String(value);
  return /[",\r\n]/.test(rendered) ? `"${rendered.replaceAll('"', '""')}"` : rendered;
}

function toCsv(matrix) {
  return matrix.map((row) => row.map(csvEscape).join(",")).join("\r\n") + "\r\n";
}

await fs.mkdir(intermediate, { recursive: true });
const workbook = Workbook.create();

for (const spec of specs) {
  const records = JSON.parse(await fs.readFile(path.join(intermediate, spec.json), "utf8"));
  const columns = records.length ? Object.keys(records[0]) : [];
  if (!columns.length) throw new Error(`${spec.json} has no columns`);
  const matrix = [columns, ...records.map((record) => columns.map((column) => record[column] ?? null))];
  const sheet = workbook.worksheets.add(spec.sheet);
  const lastColumn = columnName(columns.length - 1);
  const used = sheet.getRange(`A1:${lastColumn}${matrix.length}`);
  used.values = matrix;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(3, columns.length));
  used.format.font = { name: "Aptos", size: 10 };
  used.format.verticalAlignment = "top";
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#17365D" },
  };
  header.format.rowHeight = 34;
  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    const range = sheet.getRange(`${columnName(index)}1:${columnName(index)}${matrix.length}`);
    if (column.includes("date") || column.endsWith("_start") || column.endsWith("_end")) {
      range.format.columnWidth = 14;
    } else if (["ADV20"].includes(column)) {
      range.format.columnWidth = 18;
      range.format.numberFormat = "$#,##0";
    } else if (column.includes("months")) {
      range.format.columnWidth = 16;
      range.format.numberFormat = "0.00";
    } else if (["classification_evidence", "historical_classification_rule", "ineligibility_reason", "rejection_reason", "unresolved_fields", "reason_needs_review", "details"].includes(column)) {
      range.format.columnWidth = 42;
      range.format.wrapText = true;
    } else if (["source", "classification_source"].includes(column)) {
      range.format.columnWidth = 44;
    } else if (column === "fund_name") {
      range.format.columnWidth = 34;
      range.format.wrapText = true;
    } else if (column.includes("ticker")) {
      range.format.columnWidth = column === "ticker" ? 12 : 34;
    } else {
      range.format.columnWidth = Math.min(24, Math.max(12, column.length + 2));
    }
  }
  if (matrix.length > 1) {
    sheet.tables.add(`A1:${lastColumn}${matrix.length}`, true, `${spec.sheet.replaceAll(" ", "")}Table`);
  }
  const authored = sheet.getRange(`A1:${lastColumn}${matrix.length}`).values;
  const outputPath = path.join(root, ...spec.output);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, toCsv(authored), "utf8");

  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: spec.sheet,
    range: `A1:${lastColumn}${Math.min(matrix.length, 8)}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: Math.min(columns.length, 18),
    maxChars: 2500,
  });
  console.log(`${path.basename(outputPath)}: ${records.length} data rows`);
  console.log(inspect.ndjson);

  const preview = await workbook.render({
    sheetName: spec.sheet,
    range: `A1:${lastColumn}${Math.min(matrix.length, 25)}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(intermediate, `${path.basename(outputPath)}.preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "Step 2 formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(intermediate, "step2_sector_universe_validation.xlsx"));
