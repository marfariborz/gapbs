// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details
#include <emmintrin.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <string.h>
#include <smmintrin.h> 

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. It perform
updates in the pull direction to remove the need for atomics, and it allows
new values to be immediately visible (like Gauss-Seidel method). The prior PR
implemention is still available in src/pr_spmv.cc.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;


void force_nt_load(NodeID *p) {
  _mm_prefetch (p, _MM_HINT_NTA);
}

__m128i force_nt_store(__m128i zeros, NodeID *a) {
    __asm volatile("MOVNTDQA (%1), %0\n\t"
                    "MOVNTDQA 16(%1), %0\n\t"
                    "MOVNTDQA 32(%1), %0\n\t"
                    "MOVNTDQA 48(%1), %0\n\t"
                   :: "x" (zeros), "r" (&a): "memory");
    return zeros;
}

// void force_nt_store(__m128i a, NodeID *v) {
//     // do 4 stores to hit whole cache line
//     __asm volatile("movntdq %0, (%1)\n\t"
//                    "movntdq %0, 16(%1)\n\t"
//                    "movntdq %0, 32(%1)\n\t"
//                    "movntdq %0, 48(%1)"
//                    :
//                    : "x" (a), "r" (&v)
//                    : "memory");
// }

pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters,
                             double epsilon = 0) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  // const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  // pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> incoming_total(g.num_nodes(), 0);
  pvector<ScoreT> outgoing_contrib(g.num_nodes()/2);
  pvector<ScoreT> outgoing_contrib_n(g.num_nodes()/2);
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes()/2; n++)
    outgoing_contrib[n] = init_score / g.out_degree(n);
  for (NodeID n=g.num_nodes()/2; n < g.num_nodes(); n++)
    outgoing_contrib_n[n - g.num_nodes()/2] = init_score / g.out_degree(n);
  for (int iter=0; iter < max_iters; iter++) {
    // double error = 0;
    #pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) {
      NodeID* itr;
      for (itr = g.in_neigh(u).begin(); itr < g.in_neigh(u).end(); itr++){
        __m128i in = {0, 0};
        __m128i var = force_nt_store(in, itr);
        NodeID v = (NodeID)var[0];
        if (v < g.num_nodes()/2){
          incoming_total[u] += outgoing_contrib[v];
        } else break;
      }
    }
    #pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) {
      NodeID* itr;
      for (itr = g.in_neigh(u).begin(); itr < g.in_neigh(u).end(); itr++){
        __m128i in = {0, 0};
        __m128i var = force_nt_store(in, itr);
        NodeID v = (NodeID)var[0];
        if ((g.num_nodes()/2) <= v)
          incoming_total[u] += outgoing_contrib_n[v];
      }
      // ScoreT old_score = scores[u];
      // scores[u] = base_score + kDamp * incoming_total[u];
      // error += fabs(incoming_total[u] - outgoing_contrib[u]);
      // outgoing_contrib[u] = scores[u] / g.out_degree(u);
      // incoming_total[u] = 0;
    }
    printf(" %2d\n", iter);
    // if (error < epsilon)
    //   break;
  }
  return incoming_total;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  auto PRBound = [&cli] (const Graph &g) {
    return PageRankPullGS(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
