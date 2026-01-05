function calculateMAPE(actual, forecast) {
  const n = actual.length;
  const sumAPE = actual.reduce((acc, val, i) => {
    if (val === 0) return acc;
    return acc + Math.abs((val - forecast[i]) / val);
  }, 0);
  return (sumAPE / n) * 100;
}

const actualInput = process.argv[2];
const forecastInput = process.argv[3];

if (!actualInput || !forecastInput) {
  console.log("วิธีใช้: node mape.js <ค่าจริง> <ค่าพยากรณ์>");
  console.log("ตัวอย่าง: node mape.js 100,200,150 90,220,135");
  process.exit(1);
}

const actual = actualInput.split(",").map(Number);
const forecast = forecastInput.split(",").map(Number);

if (actual.length !== forecast.length) {
  console.error("Error: จำนวนข้อมูลไม่เท่ากัน!");
} else {
  const result = calculateMAPE(actual, forecast);
  console.log("----------------------------");
  console.log(`จำนวนข้อมูล (n): ${actual.length}`);
  
  let colorStart = "";
  let colorEnd = "\x1b[0m";
  if (result < 5) {
    colorStart = "\x1b[32m"; 
  } else if (result < 11) {
    colorStart = "\x1b[33m"; 
  } else {
    colorStart = "\x1b[31m"; 
  }
  console.log(`${colorStart}MAPE: ${result.toFixed(2)}%${colorEnd}`);
  console.log("----------------------------");
}
