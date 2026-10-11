const sign = (x: number) => (x > 0 ? "+" : x < 0 ? "−" : "");

/** +$18.4k, −$3.1k, $950. Always signed when `signed`. */
export function money(x: number, signed = true): string {
  const abs = Math.abs(x);
  const body =
    abs >= 1_000_000
      ? `${(abs / 1_000_000).toFixed(2)}M`
      : abs >= 10_000
        ? `${(abs / 1000).toFixed(1)}k`
        : abs >= 1000
          ? `${(abs / 1000).toFixed(2)}k`
          : abs.toFixed(0);
  return `${signed ? sign(x) : x < 0 ? "−" : ""}$${body}`;
}

export function units(x: number, signed = true): string {
  const abs = Math.abs(x);
  const body = abs >= 10_000 ? `${(abs / 1000).toFixed(1)}k` : Math.round(abs).toLocaleString("en-US");
  return `${signed ? sign(x) : x < 0 ? "−" : ""}${body}`;
}

export const percent = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

/** A price: always two decimals, e.g. $19.20. */
export const dollars = (x: number) => `$${x.toFixed(2)}`;
