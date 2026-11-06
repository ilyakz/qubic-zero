/* Fast part of the rules: what the search calls on every new position.
 * Same logic as find_vcf_py in bots.py; the tests check that both versions agree.
 * Built automatically by qubic.fast, once for every board width (-DWORDS=1, 2, 4...). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef WORDS
#define WORDS 1
#endif
#define SEEN_SIZE 4096

enum { CODE_NONE = 0, CODE_WIN = 1, CODE_DRAW = 2, CODE_LOST = 3, CODE_FORCED = 4, CODE_VCF = 5 };

typedef struct { uint64_t w[WORDS]; } board;

typedef struct {
    int n_lines, line_len, cells;
    board *lines;
    int *order; /* cells from the most lines to the fewest: candidates are tried in this order */
    board seen_mine[SEEN_SIZE], seen_theirs[SEEN_SIZE];
    uint32_t seen_gen[SEEN_SIZE], generation;
} ctx;

static inline board b_load(const void *p) { board b; memcpy(b.w, p, sizeof b.w); return b; }
static inline board b_or(board a, board b) { for (int i = 0; i < WORDS; i++) a.w[i] |= b.w[i]; return a; }
static inline board b_and(board a, board b) { for (int i = 0; i < WORDS; i++) a.w[i] &= b.w[i]; return a; }
static inline board b_andnot(board a, board b) { for (int i = 0; i < WORDS; i++) a.w[i] &= ~b.w[i]; return a; }
static inline int b_any(board a) { for (int i = 0; i < WORDS; i++) if (a.w[i]) return 1; return 0; }
static inline int b_hits(board a, board b) { for (int i = 0; i < WORDS; i++) if (a.w[i] & b.w[i]) return 1; return 0; }
static inline int b_eq(board a, board b) { for (int i = 0; i < WORDS; i++) if (a.w[i] != b.w[i]) return 0; return 1; }
static inline int b_count(board a) { int n = 0; for (int i = 0; i < WORDS; i++) n += __builtin_popcountll(a.w[i]); return n; }
static inline board b_bit(int cell) { board b; memset(b.w, 0, sizeof b.w); b.w[cell >> 6] = 1ULL << (cell & 63); return b; }

/* Exactly one bit set. */
static inline int b_one(board a) {
    int seen = 0;
    for (int i = 0; i < WORDS; i++) {
        uint64_t v = a.w[i];
        if (!v) continue;
        if (seen || (v & (v - 1))) return 0;
        seen = 1;
    }
    return seen;
}

/* Lowest set bit, or -1. */
static inline int b_first(board a) {
    for (int i = 0; i < WORDS; i++) if (a.w[i]) return i * 64 + __builtin_ctzll(a.w[i]);
    return -1;
}

void *qubic_new(int n_lines, int line_len, int cells, const char *masks) {
    ctx *c = calloc(1, sizeof(ctx));
    c->n_lines = n_lines;
    c->line_len = line_len;
    c->cells = cells;
    c->lines = malloc(sizeof(board) * n_lines);
    int *degree = calloc(cells, sizeof(int));
    int top = 0;
    for (int i = 0; i < n_lines; i++) {
        c->lines[i] = b_load(masks + i * sizeof(board));
        for (int w = 0; w < WORDS; w++)
            for (uint64_t m = c->lines[i].w[w]; m; m &= m - 1) {
                int d = ++degree[w * 64 + __builtin_ctzll(m)];
                if (d > top) top = d;
            }
    }
    c->order = malloc(sizeof(int) * cells);
    int filled = 0;
    for (int d = top; d >= 0; d--)
        for (int cell = 0; cell < cells; cell++)
            if (degree[cell] == d) c->order[filled++] = cell;
    free(degree);
    return c;
}

/* Cells where own wins at once. */
static inline board wins(const ctx *c, board own, board opp) {
    board result;
    memset(result.w, 0, sizeof result.w);
    for (int i = 0; i < c->n_lines; i++) {
        board m = c->lines[i];
        if (b_hits(m, opp)) continue;
        board rest = b_andnot(m, own);
        if (b_one(rest)) result = b_or(result, rest);
    }
    return result;
}

/* 1 if this pair of positions was already seen in the current search. */
static inline int mark_seen(ctx *c, board mine, board theirs) {
    uint64_t h = 0;
    for (int i = 0; i < WORDS; i++) h = (h ^ mine.w[i] ^ (theirs.w[i] * 0x9E3779B97F4A7C15ULL)) * 0xC2B2AE3D27D4EB4FULL;
    uint32_t i = (uint32_t)(h >> 52);
    for (int probe = 0; probe < 8; probe++, i = (i + 1) & (SEEN_SIZE - 1)) {
        if (c->seen_gen[i] != c->generation) {
            c->seen_gen[i] = c->generation;
            c->seen_mine[i] = mine;
            c->seen_theirs[i] = theirs;
            return 0;
        }
        if (b_eq(c->seen_mine[i], mine) && b_eq(c->seen_theirs[i], theirs)) return 1;
    }
    return 0;
}

static int search(ctx *c, board own, board opp, int depth, int *left) {
    if (--*left < 0) return -1;
    board w = wins(c, own, opp);
    if (b_any(w)) return b_first(w);
    board threats = wins(c, opp, own);
    int n_threats = b_count(threats);
    if (n_threats >= 2 || depth == 0) return -1;
    board candidates;
    memset(candidates.w, 0, sizeof candidates.w);
    for (int i = 0; i < c->n_lines; i++) {
        board m = c->lines[i];
        if (!b_hits(m, opp) && b_count(b_and(m, own)) == c->line_len - 2)
            candidates = b_or(candidates, b_andnot(m, own));
    }
    if (n_threats) candidates = b_and(candidates, threats);
    if (!b_any(candidates)) return -1;
    for (int k = 0; k < c->cells; k++) {
        int cell = c->order[k];
        if (!b_hits(candidates, b_bit(cell))) continue;
        board mine = b_or(own, b_bit(cell));
        board replies = wins(c, mine, opp);
        int n_replies = b_count(replies);
        if (n_replies >= 2) return cell;
        if (!n_replies) continue;
        board theirs = b_or(opp, replies);
        if (mark_seen(c, mine, theirs)) continue;
        if (search(c, mine, theirs, depth - 1, left) >= 0) return cell;
    }
    return -1;
}

int qubic_find_vcf(void *handle, const char *own_bytes, const char *opp_bytes, int budget, int depth) {
    ctx *c = handle;
    c->generation++;
    int left = budget;
    return search(c, b_load(own_bytes), b_load(opp_bytes), depth, &left);
}

/* What follows from the rules without the network: code | cell << 8. */
int qubic_analyse(void *handle, const char *own_bytes, const char *opp_bytes, int budget) {
    ctx *c = handle;
    board own = b_load(own_bytes), opp = b_load(opp_bytes);
    board w = wins(c, own, opp);
    if (b_any(w)) return CODE_WIN | b_first(w) << 8;
    if (b_count(b_or(own, opp)) == c->cells) return CODE_DRAW;
    board threats = wins(c, opp, own);
    int n_threats = b_count(threats);
    if (n_threats >= 2) return CODE_LOST | b_first(threats) << 8;
    if (budget > 0) {
        c->generation++;
        int left = budget;
        int cell = search(c, own, opp, 16, &left);
        if (cell >= 0) return CODE_VCF | cell << 8;
    }
    if (n_threats) return CODE_FORCED | b_first(threats) << 8;
    return CODE_NONE;
}
