#!/usr/bin/env node
/**
 * Parse Auction Express–style Excel into normalized JSON.
 * - Skips first "SUM" sheet
 * - Processes only digit-like sheet names (e.g., 06012025, 6102025)
 * - Detects English + Thai headers
 * - Removes Thai "note" rows (หมายเหตุ/ผู้ซื้อ/ภาษีหมด/เช็คต้น/ต่อภาษี)
 * - Outputs array of objects with required columns
 *
 * Usage:
 *   npm i xlsx
 *   node parse-auction.js "Auction Express 2025.xlsx" \
 *     --out auction_express_parsed_clean.json \
 *     --sourceName "Auction Express" \
 *     --sourceType "Auction" \
 *     --sourceUrl "Not Applicable"
 */

const fs = require('fs');
const path = require('path');
const XLSX = require('xlsx');

const NOT_APPL = 'Not Applicable';

const HEADER_ALIASES = {
  brand: ['brand','make','manufacturer','ยี่ห้อ'],
  model: ['model','รุ่น'],
  submodel: ['submodel','trim','variant','sub model','sub-model','รุ่นย่อย'],
  year: ['year','yr','model year','mfg year','ปีที่ผลิต','ปีที่จดทะเบียน'],
  gear: ['gear','transmission','gearbox','ระบบเกียร์'],
  engine: ['engine','engine size','engine displacement','displacement','ความจุ','cc','ความจุ (cc)'],
  color: ['color','colour','exterior color','body color','สี'],
  mileage: ['mileage','odo','odometer','km','kms','kilometers','kilometres','miles','mile','เลขไมล์','เลขไมล์ (กม.)'],
  price: ['price','current bid','currentbid','buy now','buynow','final price','hammer price','bid','ราคาปัจจุบัน','ราคาเริ่มต้น','ราคาสูงสุด','ราคาเริ่มต้น (ไม่รวมvat)'],
  externalId: ['lot no','lot','lot number','id','stock no','stock number','ref','reference','auction id','ลำดับที่'],
};

const NOTE_KEYWORDS = ['หมายเหตุ','ผู้ซื้อ','ภาษีหมด','เช็คต้น','ต่อภาษี'];

function getArg(flag, fallback) {
  const i = process.argv.indexOf(flag);
  return i !== -1 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const inFile = process.argv[2];
if (!inFile) {
  console.error('Usage: node parse-auction.js "<file.xlsx>" [--out out.json] [--sourceName ...] [--sourceType ...] [--sourceUrl ...]');
  process.exit(1);
}
const outFile = getArg('--out', 'parsed.json');
const sourceNameDefault = getArg('--sourceName', 'Auction Express');
const sourceTypeDefault = getArg('--sourceType', 'Auction');
const sourceUrlDefault = getArg('--sourceUrl', 'Not Applicable');

function onlyDigits(s) { return String(s || '').replace(/\D+/g, ''); }

function parseDateFromSheetName(name) {
  const d = onlyDigits(name);
  const m = d.match(/^(\d{1,2})(\d{1,2})(\d{4})$/);
  if (!m) return NOT_APPL;
  const day = Number(m[1]), month = Number(m[2]), year = Number(m[3]);
  if (year < 1900 || month < 1 || month > 12 || day < 1 || day > 31) return NOT_APPL;
  return `${year}-${String(month).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
}

function normalizeIntegerLike(value) {
  if (value == null) return null;
  const s = String(value).trim();
  if (!s) return null;
  const digits = s.replace(/[^\d]/g, '');
  return digits ? Number(digits) : null;
}

function normalizeNumberLike(value) {
  if (value == null) return null;
  let s = String(value).trim();
  if (!s) return null;
  s = s.replace(/[^0-9.\-]/g, '');
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function inferPriceType(usedHeader) {
  if (!usedHeader) return NOT_APPL;
  const h = usedHeader.toLowerCase();
  if (h.includes('current') || h.includes('ปัจจุบัน')) return 'CurrentBid';
  if (h.includes('buy')) return 'BuyNow';
  if (h.includes('hammer') || h.includes('final') || h.includes('สูงสุด')) return 'HammerPrice';
  if (h.includes('เริ่มต้น')) return 'StartingPrice';
  if (h.includes('bid')) return 'Bid';
  return 'Price';
}

function locateHeaderRow(sheet) {
  const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, raw: false, defval: '' });
  const keys = new Set(['ยี่ห้อ','รุ่น','รุ่นย่อย','ปีที่ผลิต','เลขไมล์','ระบบเกียร์','สี','ราคาปัจจุบัน','ราคาเริ่มต้น','ราคาสูงสุด','ลำดับที่']);
  const maxLook = Math.min(10, rows.length);
  for (let i = 0; i < maxLook; i++) {
    const rowVals = new Set((rows[i] || []).map(v => String(v || '').trim()).filter(Boolean));
    for (const k of rowVals) {
      if (keys.has(k)) return i;
    }
  }
  return 0;
}

function findHeaderKey(headerRow, target) {
  const aliases = HEADER_ALIASES[target] || [];
  const lowerMap = {};
  headerRow.forEach((h, idx) => {
    if (h == null) return;
    lowerMap[String(h).trim().toLowerCase()] = idx;
  });
  for (const alias of aliases) {
    const a = alias.toLowerCase();
    if (a in lowerMap) return { index: lowerMap[a], headerName: alias };
    for (const key in lowerMap) {
      if (key.includes(a)) return { index: lowerMap[key], headerName: key };
    }
  }
  return null;
}

function pick(arr, i) { return i == null ? null : (arr[i] ?? null); }

const wb = XLSX.readFile(inFile, { cellDates: false });
const sheetNames = wb.SheetNames || [];
const results = [];

sheetNames.forEach((sheetName, idx) => {
  if (idx === 0 && String(sheetName).trim().toUpperCase() === 'SUM') return;

  const digits = onlyDigits(sheetName);
  if (!digits || digits.length < 6) return;

  const ws = wb.Sheets[sheetName];
  if (!ws) return;

  const all = XLSX.utils.sheet_to_json(ws, { header: 1, raw: false, defval: '' });
  if (!all.length) return;

  const headerIdx = locateHeaderRow(ws);
  const headerRow = (all[headerIdx] || []).map(h => String(h || '').trim());
  const dataRows = all.slice(headerIdx + 1);

  const idxBrand = findHeaderKey(headerRow, 'brand');
  const idxModel = findHeaderKey(headerRow, 'model');
  const idxSubmodel = findHeaderKey(headerRow, 'submodel');
  const idxYear = findHeaderKey(headerRow, 'year');
  const idxGear = findHeaderKey(headerRow, 'gear');
  const idxEngine = findHeaderKey(headerRow, 'engine');
  const idxColor = findHeaderKey(headerRow, 'color');
  const idxMileage = findHeaderKey(headerRow, 'mileage');
  const idxPrice = findHeaderKey(headerRow, 'price');
  const idxExternalId = findHeaderKey(headerRow, 'externalId');

  const priceHeaderUsed = idxPrice ? idxPrice.headerName : null;
  const dateRecorded = parseDateFromSheetName(sheetName);

  for (const row of dataRows) {
    const brand = (pick(row, idxBrand && idxBrand.index) || '').trim();
    const model = (pick(row, idxModel && idxModel.index) || '').trim();
    const submodel = (pick(row, idxSubmodel && idxSubmodel.index) || '').trim();
    let year = normalizeIntegerLike(pick(row, idxYear && idxYear.index));
    if (year != null) {
      if (year < 100 && year > 30) year = 1900 + year;
      else if (year < 30) year = 2000 + year;
    }
    const gear = (pick(row, idxGear && idxGear.index) || '').trim();
    const engine = (pick(row, idxEngine && idxEngine.index) || '').trim();
    const color = (pick(row, idxColor && idxColor.index) || '').trim();
    const mileage = normalizeIntegerLike(pick(row, idxMileage && idxMileage.index));
    const price = normalizeNumberLike(pick(row, idxPrice && idxPrice.index));
    const priceType = price != null ? inferPriceType(priceHeaderUsed) : NOT_APPL;
    const externalId = (pick(row, idxExternalId && idxExternalId.index) || '').trim();

    // ----- NOTE FILTER -----
    const rowText = JSON.stringify(row).toLowerCase();
    const isNote = NOTE_KEYWORDS.some(k => rowText.includes(k.toLowerCase()));

    const rec = {
      brand: brand || NOT_APPL,
      model: model || NOT_APPL,
      submodel: submodel || NOT_APPL,
      year: Number.isInteger(year) ? year : NOT_APPL,
      gear: gear || NOT_APPL,
      engine: engine || NOT_APPL,
      color: color || NOT_APPL,
      mileage: Number.isInteger(mileage) ? mileage : NOT_APPL,
      price: typeof price === 'number' && Number.isFinite(price) ? price : NOT_APPL,
      priceType,
      sourceName: sourceNameDefault || NOT_APPL,
      sourceType: sourceTypeDefault || NOT_APPL,
      sourceUrl: sourceUrlDefault || NOT_APPL,
      externalId: externalId || NOT_APPL,
      dateRecorded: dateRecorded || NOT_APPL,
    };

    const noCoreFields = rec.brand === NOT_APPL && rec.model === NOT_APPL && rec.externalId === NOT_APPL;

    if (!isNote && !noCoreFields) {
      results.push(rec);
    }
  }
});

fs.writeFileSync(outFile, JSON.stringify(results, null, 2), 'utf8');
console.log(`✅ Wrote ${results.length} records to ${path.resolve(outFile)}`);
