import { test } from "node:test";
import assert from "node:assert/strict";
import { RULES, tokens } from "../src/core.js";
import { MemoryStore } from "../src/stores/memory.js";
import { rowsOf } from "../src/stores/hydradb.js";

test("alias rules produce the documented aliases", () => {
  assert.deepEqual(RULES.paren.alias("Ed Wood (film)"), ["ed wood"]);
  assert.deepEqual(RULES.the.alias("The Beatles"), ["beatles"]);
  assert.deepEqual(RULES.surname.alias("Scott Derrickson"), ["derrickson"]);
  assert.deepEqual(RULES.surname.alias("the art of war"), []);
  assert.deepEqual(RULES.acronym.alias("Federal Bureau of Investigation"), ["fbi"]);
  assert.deepEqual(RULES.comma.alias("Paris, Texas"), ["paris"]);
  assert.deepEqual(tokens("Café Müller's"), ["cafe", "muller", "s"]);
});

test("memory store: pairs count once, rules roll back cleanly", async () => {
  const s = new MemoryStore();
  assert.equal(await s.addEdges([[1, 2, "exact"], [2, 3, "exact"]]), 2);
  assert.equal(await s.addEdges([[1, 2, "paren"], [3, 4, "paren"]]), 1);
  assert.equal(await s.edgeCount(), 3);
  await s.removeRule("paren");
  assert.equal(await s.edgeCount(), 2);
  const nb = await s.neighbors([2]);
  assert.deepEqual(nb.out.get(2), [3]);
  assert.deepEqual(nb.in.get(2), [1]);
});

test("HydraDB HTTP rows are unwrapped from tagged values", () => {
  const body = { columns: ["a", "b", "l"], rows: [[{ type: "integer", value: 7 }, { type: "string", value: "x" }, { type: "list", value: [{ type: "integer", value: 1 }] }]], next_cursor: null };
  assert.deepEqual(rowsOf(body), [{ a: 7, b: "x", l: [1] }]);
});
