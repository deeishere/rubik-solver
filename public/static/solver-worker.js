// Kociemba two-phase solver running off the main thread (port of app.py's /api/solve).
// Bump ?v= whenever the vendor files change; they are cached for a week (see _headers).
importScripts('vendor/cube.js?v=2', 'vendor/solve.js?v=2');

const FACES = 'URFDLB';
let ready = false;

function parity(perm) {
  let p = 0;
  for (let i = 0; i < perm.length; i++) for (let j = i + 1; j < perm.length; j++) if (perm[i] > perm[j]) p ^= 1;
  return p;
}

function validate(state) {
  if (state.length !== 54 || /[^URFDLB]/.test(state)) return 'Enter exactly 54 face letters: U, R, F, D, L, B.';
  for (const f of FACES) if (state.split(f).length - 1 !== 9) return 'Each face color must appear exactly nine times.';
  for (let i = 0; i < 6; i++) if (state[i * 9 + 4] !== FACES[i]) return `The center of face ${FACES[i]} must be ${FACES[i]}. Check its colors.`;
  const cube = Cube.fromString(state);
  const bad = 'These stickers do not form a valid cube: ';
  if (cube.asString() !== state) return bad + 'some corner or edge has an impossible color combination.';
  if (new Set(cube.cp).size !== 8) return bad + 'a corner piece appears twice.';
  if (new Set(cube.ep).size !== 12) return bad + 'an edge piece appears twice.';
  if (cube.co.reduce((a, b) => a + b, 0) % 3) return bad + 'a corner is twisted.';
  if (cube.eo.reduce((a, b) => a + b, 0) % 2) return bad + 'an edge is flipped.';
  if (parity(cube.cp) !== parity(cube.ep)) return bad + 'two pieces are swapped.';
  return null;
}

// The two-phase search returns the first solution it finds, which can be far
// from the shortest (e.g. 9 moves for a 2-move scramble). Keep asking for a
// shorter one until a search fails or the time budget runs out.
function shortestSolve(cube, budgetMs = 2000) {
  const deadline = Date.now() + budgetMs;
  let best = cube.solve().trim();
  while (Date.now() < deadline) {
    const length = best.split(/\s+/).length;
    if (length <= 1) break;
    let shorter;
    try { shorter = cube.solve(length - 1, deadline); } catch (e) { if (e.message === 'timeout') break; throw e; }
    if (!shorter) break; // no solution that short exists within this search
    best = shorter.trim();
  }
  return best;
}

self.onmessage = ({ data }) => {
  if (data.cmd === 'init') {
    if (!ready) { Cube.initSolver(); ready = true; }
    self.postMessage({ cmd: 'init' });
    return;
  }
  if (data.cmd === 'solve') {
    if (!ready) { Cube.initSolver(); ready = true; }
    const error = validate(data.state);
    if (error) { self.postMessage({ cmd: 'solve', id: data.id, error }); return; }
    const cube = Cube.fromString(data.state);
    const solution = cube.isSolved() ? '' : shortestSolve(cube);
    self.postMessage({ cmd: 'solve', id: data.id, solution, moves: solution ? solution.split(/\s+/) : [] });
  }
};
