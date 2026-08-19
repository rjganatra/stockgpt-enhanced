/**
 * A small, self-contained parser/evaluator for the subset of pandas
 * `.query()` syntax this app actually uses -- comparisons, `and`/`or`/`not`,
 * `in [...]`, and `.str.contains('x')` -- so Strategy Lab's entry/exit
 * conditions can be evaluated in the browser exactly the same way
 * `panel.eval(strategy.entry_query)` evaluates them in
 * src/stockgpt/backtest/engine.py, without needing to ship a Python
 * runtime to do it.
 *
 * Why not just JS `eval()`: that would execute arbitrary code with no
 * sandboxing (a real risk once a visitor's own typed text reaches it), and
 * it doesn't understand Python's `and`/`or`/`in` keywords or comparison
 * semantics (`NaN >= 65` happens to agree between JS and pandas/numpy, but
 * `and`/`or`/`in` are Python-only syntax `eval()` would just throw on).
 * A small dedicated grammar is both safer and more correct for this one job.
 *
 * Missing-value semantics deliberately mirror pandas/numpy: a numeric
 * column's missing value must be represented as JS `NaN` (not null/
 * undefined) so that `NaN >= x`, `NaN <= x`, `NaN == x` all evaluate false
 * and `NaN != x` evaluates true -- IEEE-754 already behaves exactly like
 * numpy here, no special-casing needed. A missing string/categorical value
 * should be represented as `null`; every comparison against `null` here
 * evaluates false (conservative -- "we don't know" should never silently
 * match a filter).
 */

class QueryParseError extends Error {}

function tokenize(expr) {
  const tokens = [];
  let i = 0;
  const n = expr.length;
  const isIdentStart = (c) => /[A-Za-z_]/.test(c);
  const isIdentChar = (c) => /[A-Za-z0-9_]/.test(c);
  const isDigit = (c) => /[0-9]/.test(c);

  while (i < n) {
    const c = expr[i];
    if (c === " " || c === "\t" || c === "\n") { i++; continue; }

    if (c === "(" || c === ")" || c === "[" || c === "]" || c === "," || c === ".") {
      tokens.push({ type: c, value: c });
      i++;
      continue;
    }

    // Two-character operators first, so ">=" isn't tokenized as ">" then "=".
    const two = expr.slice(i, i + 2);
    if (two === "==" || two === "!=" || two === ">=" || two === "<=" || two === "&&" || two === "||") {
      tokens.push({ type: "OP", value: two === "&&" ? "and" : two === "||" ? "or" : two });
      i += 2;
      continue;
    }
    if (c === ">" || c === "<") {
      tokens.push({ type: "OP", value: c });
      i++;
      continue;
    }
    if (c === "!") {
      tokens.push({ type: "NOT", value: "not" });
      i++;
      continue;
    }

    if (c === "'" || c === '"') {
      const quote = c;
      let j = i + 1;
      let out = "";
      while (j < n && expr[j] !== quote) {
        out += expr[j];
        j++;
      }
      if (j >= n) throw new QueryParseError(`Unterminated string literal starting at position ${i}`);
      tokens.push({ type: "STRING", value: out });
      i = j + 1;
      continue;
    }

    if (isDigit(c) || (c === "-" && isDigit(expr[i + 1] || ""))) {
      let j = i + (c === "-" ? 1 : 0);
      while (j < n && isDigit(expr[j])) j++;
      if (expr[j] === ".") {
        j++;
        while (j < n && isDigit(expr[j])) j++;
      }
      tokens.push({ type: "NUMBER", value: parseFloat(expr.slice(i, j)) });
      i = j;
      continue;
    }
    if (c === "-") {
      tokens.push({ type: "OP", value: "-unary" });
      i++;
      continue;
    }

    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentChar(expr[j])) j++;
      const word = expr.slice(i, j);
      const lower = word.toLowerCase();
      if (lower === "and") tokens.push({ type: "AND", value: "and" });
      else if (lower === "or") tokens.push({ type: "OR", value: "or" });
      else if (lower === "not") tokens.push({ type: "NOT", value: "not" });
      else if (lower === "in") tokens.push({ type: "IN", value: "in" });
      else tokens.push({ type: "IDENT", value: word });
      i = j;
      continue;
    }

    throw new QueryParseError(`Unexpected character '${c}' at position ${i} in: ${expr}`);
  }
  tokens.push({ type: "EOF", value: null });
  return tokens;
}

class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }
  peek() { return this.tokens[this.pos]; }
  next() { return this.tokens[this.pos++]; }
  expect(type) {
    const t = this.next();
    if (t.type !== type) throw new QueryParseError(`Expected ${type} but got ${t.type} ('${t.value}')`);
    return t;
  }

  parseExpr() { return this.parseOr(); }

  parseOr() {
    let left = this.parseAnd();
    while (this.peek().type === "OR") {
      this.next();
      const right = this.parseAnd();
      left = { kind: "or", left, right };
    }
    return left;
  }

  parseAnd() {
    let left = this.parseNot();
    while (this.peek().type === "AND") {
      this.next();
      const right = this.parseNot();
      left = { kind: "and", left, right };
    }
    return left;
  }

  parseNot() {
    if (this.peek().type === "NOT") {
      this.next();
      return { kind: "not", operand: this.parseNot() };
    }
    return this.parseComparison();
  }

  parseComparison() {
    const left = this.parsePostfix();
    const t = this.peek();
    if (t.type === "OP" && ["==", "!=", ">=", "<=", ">", "<"].includes(t.value)) {
      this.next();
      const right = this.parsePostfix();
      return { kind: "compare", op: t.value, left, right };
    }
    if (t.type === "IN") {
      this.next();
      const list = this.parseList();
      return { kind: "in", left, list };
    }
    return left;
  }

  parseList() {
    this.expect("[");
    const items = [];
    if (this.peek().type !== "]") {
      items.push(this.parsePrimary());
      while (this.peek().type === ",") {
        this.next();
        items.push(this.parsePrimary());
      }
    }
    this.expect("]");
    return items;
  }

  // Postfix handles `field.str.contains('x')`. Not general attribute
  // access -- this app only ever needs that one pandas idiom (see
  // render_column_reference's advertised text-field syntax), so that's all
  // this parses; anything else after a `.` is a hard error, not silently
  // ignored.
  parsePostfix() {
    let node = this.parsePrimary();
    while (this.peek().type === ".") {
      this.next();
      const word = this.expect("IDENT").value;
      if (word !== "str") throw new QueryParseError(`Unsupported attribute '.${word}' (only '.str.contains(...)' is supported)`);
      this.expect(".");
      const method = this.expect("IDENT").value;
      if (method !== "contains") throw new QueryParseError(`Unsupported string method '.${method}' (only '.contains(...)' is supported)`);
      this.expect("(");
      const arg = this.parsePrimary();
      this.expect(")");
      node = { kind: "str_contains", target: node, arg };
    }
    return node;
  }

  parsePrimary() {
    const t = this.peek();
    if (t.type === "NUMBER") { this.next(); return { kind: "literal", value: t.value }; }
    if (t.type === "STRING") { this.next(); return { kind: "literal", value: t.value }; }
    if (t.type === "IDENT") { this.next(); return { kind: "field", name: t.value }; }
    if (t.type === "OP" && t.value === "-unary") {
      this.next();
      const operand = this.parsePrimary();
      if (operand.kind !== "literal" || typeof operand.value !== "number") {
        throw new QueryParseError("Unary '-' is only supported directly on a numeric literal");
      }
      return { kind: "literal", value: -operand.value };
    }
    if (t.type === "(") {
      this.next();
      const inner = this.parseExpr();
      this.expect(")");
      return inner;
    }
    throw new QueryParseError(`Unexpected token ${t.type} ('${t.value}')`);
  }
}

function parseQuery(expr) {
  const tokens = tokenize(expr);
  const parser = new Parser(tokens);
  const ast = parser.parseExpr();
  if (parser.peek().type !== "EOF") {
    throw new QueryParseError(`Unexpected trailing input near '${parser.peek().value}'`);
  }
  return ast;
}

/**
 * Evaluates a literal/field/str_contains node against row index `i` of a
 * columnar panel (`{columns, data}}`, `data[col][i]` = value). Comparison
 * and boolean nodes return true/false; literal/field nodes return the raw
 * value they represent.
 */
function evalNode(node, panel, i, knownColumns) {
  switch (node.kind) {
    case "literal":
      return node.value;
    case "field": {
      if (!knownColumns.has(node.name)) {
        throw new QueryParseError(`Unknown column '${node.name}'`);
      }
      return panel.data[node.name][i];
    }
    case "str_contains": {
      const target = evalNode(node.target, panel, i, knownColumns);
      const arg = evalNode(node.arg, panel, i, knownColumns);
      if (target === null || target === undefined) return false;
      return String(target).includes(String(arg));
    }
    case "compare": {
      const l = evalNode(node.left, panel, i, knownColumns);
      const r = evalNode(node.right, panel, i, knownColumns);
      return compareValues(node.op, l, r);
    }
    case "in": {
      const l = evalNode(node.left, panel, i, knownColumns);
      return node.list.some((item) => compareValues("==", l, evalNode(item, panel, i, knownColumns)));
    }
    case "and":
      return truthy(evalNode(node.left, panel, i, knownColumns)) && truthy(evalNode(node.right, panel, i, knownColumns));
    case "or":
      return truthy(evalNode(node.left, panel, i, knownColumns)) || truthy(evalNode(node.right, panel, i, knownColumns));
    case "not":
      return !truthy(evalNode(node.operand, panel, i, knownColumns));
    default:
      throw new QueryParseError(`Cannot evaluate node kind '${node.kind}'`);
  }
}

function truthy(v) {
  // A bare comparison/boolean node already returns a real boolean; this
  // only matters if a bare field/literal ever ends up as an and/or operand
  // (not expected from this app's queries, but fail safe rather than throw
  // deep inside a hot loop).
  if (typeof v === "boolean") return v;
  if (v === null || v === undefined) return false;
  if (typeof v === "number") return !Number.isNaN(v) && v !== 0;
  return Boolean(v);
}

function compareValues(op, l, r) {
  // NaN vs anything (including another NaN): every operator except '!='
  // must be false, matching numpy -- this happens for free in JS/IEEE-754
  // EXCEPT '==', which JS also gets right (NaN === anything is false), so
  // no special-casing needed for numbers. Strings compare by value; a
  // `null` (missing) operand on either side makes every comparator false
  // except '!=' between two non-identical values, handled below.
  if (l === null || r === null) {
    if (op === "!=") return l !== r;
    if (op === "==") return l === r;
    return false;
  }
  switch (op) {
    case "==": return l === r;
    case "!=": return l !== r;
    case ">=": return l >= r;
    case "<=": return l <= r;
    case ">": return l > r;
    case "<": return l < r;
    default: throw new QueryParseError(`Unknown comparison operator '${op}'`);
  }
}

/**
 * Evaluates `queryString` once against every row of `panel`, returning a
 * Uint8Array mask (1/0 per row) -- the JS equivalent of
 * `panel.eval(strategy.entry_query)` in engine.py, including the "evaluate
 * once across the whole panel, not per symbol" performance guarantee the
 * Python version documents.
 *
 * Throws QueryParseError (with a message meant to be shown directly to the
 * person who typed the query) on a syntax error or unknown column --
 * mirrors run_backtest's up-front validation that fails loudly on a typo
 * instead of silently returning zero signals.
 */
function evalQueryMask(panel, queryString) {
  const ast = parseQuery(queryString);
  const knownColumns = new Set(panel.columns);
  const n = panel.data[panel.columns[0]].length;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    mask[i] = truthy(evalNode(ast, panel, i, knownColumns)) ? 1 : 0;
  }
  return mask;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { parseQuery, evalQueryMask, evalNode, truthy, compareValues, QueryParseError, tokenize };
}
