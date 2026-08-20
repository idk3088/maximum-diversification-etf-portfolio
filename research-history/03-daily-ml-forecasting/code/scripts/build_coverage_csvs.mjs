import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const intermediate = path.join(root, "data", "processed", "coverage_audit_intermediate");
const outputDir = path.join(root, "results", "metadata");

const specs = [
  {
    json: "clean_universe_detail.json",
    csv: "clean_universe_detail.csv",
    sheet: "Clean Detail",
    columns: ["ticker", "fund_name", "listing_date", "first_valid_price_date", "latest_price_date", "valid_history_months", "latest_ADV20", "fund_objective", "tracked_index", "broad_market_or_sector", "likely_sector", "thematic_flag", "evidence_source", "evidence", "review_status"],
  },
  {
    json: "current_sector_coverage.json",
    csv: "current_sector_coverage.csv",
    sheet: "Current Coverage",
    columns: ["sector", "clean_candidate_count", "candidate_tickers", "earliest_available_date", "coverage_status", "missing_reason"],
  },
  {
    json: "monthly_sector_coverage_audit.json",
    csv: "monthly_sector_coverage_audit.csv",
    sheet: "Monthly Coverage",
    columns: ["month_end", "sector", "eligible_clean_etf_count", "eligible_tickers", "coverage_available", "missing_reason", "available_sector_count_in_month"],
  },
  {
    json: "priority_review_candidates.json",
    csv: "priority_review_candidates.csv",
    sheet: "Priority Review",
    columns: ["priority_rank", "target_sector", "ticker", "fund_name", "listing_date", "first_valid_price_date", "first_eligible_month_end", "latest_ADV20", "likely_asset_class", "likely_geographic_scope", "likely_sector", "unresolved_fields", "vendor_conflict_type", "official_source_available", "reason_for_priority", "evidence_source", "tracked_index"],
  },
  {
    json: "vendor_conflicts_resolved_v2.json",
    csv: "vendor_conflicts_resolved_v2.csv",
    sheet: "Vendor Conflicts V2",
    columns: ["ticker", "conflict_types", "nasdaq_fund_name", "alpha_active_names", "alpha_delisted_names", "details", "resolution_status", "normalized_nasdaq_name", "normalized_alpha_active_names", "resolution_category", "lifecycle_overlap_status", "resolved_v2", "usable_price_start_date", "resolution_evidence"],
  },
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
  const text = typeof value === "boolean" ? (value ? "True" : "False") : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(matrix) {
  return matrix.map((row) => row.map(csvEscape).join(",")).join("\r\n") + "\r\n";
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(intermediate, { recursive: true });
const workbook = Workbook.create();

for (const spec of specs) {
  const records = JSON.parse(await fs.readFile(path.join(intermediate, spec.json), "utf8"));
  const matrix = [spec.columns, ...records.map((record) => spec.columns.map((column) => record[column] ?? null))];
  const sheet = workbook.worksheets.add(spec.sheet);
  const lastColumn = columnName(spec.columns.length - 1);
  const used = sheet.getRange(`A1:${lastColumn}${matrix.length}`);
  used.values = matrix;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#17365D" },
  };
  header.format.rowHeight = 30;
  used.format.font = { name: "Aptos", size: 10 };
  used.format.verticalAlignment = "top";
  used.format.wrapText = false;
  used.format.autofitColumns();
  used.format.autofitRows();
  for (let index = 0; index < spec.columns.length; index += 1) {
    const column = spec.columns[index];
    const range = sheet.getRange(`${columnName(index)}1:${columnName(index)}${matrix.length}`);
    if (["fund_name", "fund_objective", "evidence", "missing_reason", "unresolved_fields", "reason_for_priority", "resolution_evidence"].includes(column)) {
      range.format.columnWidth = column === "fund_name" ? 32 : 42;
      range.format.wrapText = true;
    } else if (["evidence_source", "classification_sources"].includes(column)) {
      range.format.columnWidth = 45;
    } else if (column.includes("date") || column === "month_end") {
      range.format.columnWidth = 14;
    } else if (column === "latest_ADV20") {
      range.format.columnWidth = 18;
      range.format.numberFormat = "$#,##0";
    } else {
      range.format.columnWidth = Math.min(24, Math.max(12, column.length + 2));
    }
  }
  if (matrix.length > 1) {
    sheet.tables.add(`A1:${lastColumn}${matrix.length}`, true, `${spec.sheet.replaceAll(" ", "")}Table`);
  }
  const authored = sheet.getRange(`A1:${lastColumn}${matrix.length}`).values;
  await fs.writeFile(path.join(outputDir, spec.csv), toCsv(authored), "utf8");

  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: spec.sheet,
    range: `A1:${lastColumn}${Math.min(matrix.length, 8)}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: Math.min(spec.columns.length, 16),
    maxChars: 2500,
  });
  console.log(`${spec.csv}: ${records.length} data rows`);
  console.log(inspect.ndjson);

  const preview = await workbook.render({
    sheetName: spec.sheet,
    range: `A1:${lastColumn}${Math.min(matrix.length, 25)}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(intermediate, `${spec.csv}.preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "coverage audit formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(intermediate, "coverage_audit_validation.xlsx"));
