#include "vectordb/indexes/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "index_utils.hpp"

namespace vectordb {
namespace {

// A 53-bit precision uniform double in [0, 1), derived directly from the
// generator's output. std::uniform_real_distribution's sampling sequence is
// not specified byte-for-byte across standard library implementations, which
// would break cross-platform determinism for a fixed seed.
double uniform_01(std::mt19937_64 &generator) {
    constexpr int mantissa_bits = 53;
    const std::uint64_t bits = generator() >> (64 - mantissa_bits);
    return static_cast<double>(bits) /
          static_cast<double>(std::uint64_t{1} << mantissa_bits);
}

std::size_t random_level(std::mt19937_64 &generator, double inverse_ln_m) {
    double u = uniform_01(generator);
    if (u <= 0.0) {
        u = std::numeric_limits<double>::denorm_min();
    }
    return static_cast<std::size_t>(std::floor(-std::log(u) * inverse_ln_m));
}

// 1 / ln(M), the standard HNSW level-assignment scale. M == 1 would make
// ln(M) == 0; fall back to ln(2) so level assignment stays well-defined for
// that degenerate (but otherwise valid) configuration.
double level_scale(std::size_t m) {
    return 1.0 / std::log(static_cast<double>(std::max<std::size_t>(m, 2)));
}

}  // namespace

HnswIndex::HnswIndex(const VectorStore &vectors, Metric metric,
                     HnswConfig config)
    : vectors_(vectors),
      metric_(metric),
      config_(std::move(config)),
      rng_(config_.seed) {
    if (config_.M == 0) {
        throw std::invalid_argument("HNSW requires M greater than zero");
    }
    if (config_.M > HnswConfig::max_M) {
        throw std::invalid_argument(
            "HNSW M exceeds maximum supported value of " +
            std::to_string(HnswConfig::max_M));
    }
    if (config_.ef_construction == 0) {
        throw std::invalid_argument(
            "HNSW requires ef_construction greater than zero");
    }
    if (config_.ef_search == 0) {
        throw std::invalid_argument(
            "HNSW requires ef_search greater than zero");
    }
}

std::vector<HnswIndex::ScoredNode> HnswIndex::search_layer(
    std::span<const float> query,
    const std::vector<std::uint64_t> &entry_points, std::size_t ef,
    std::size_t layer,
    const std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
    bool prefer_higher) const {
    const auto orient_of = [&](std::uint64_t id) {
        const float raw = index_detail::score_vector(
            metric_, query.data(), vectors_.get(id), vectors_.dim());
        return prefer_higher ? -raw : raw;
    };

    // Min-heap by orient_score (closest first), with a deterministic
    // lower-id tie-break so a fixed seed always explores in the same order.
    const auto candidate_order = [](const ScoredNode &a, const ScoredNode &b) {
        if (a.orient_score != b.orient_score) {
            return a.orient_score > b.orient_score;
        }
        return a.id > b.id;
    };
    // Max-heap by orient_score (worst first), so the worst result is always
    // the one evicted once the set exceeds `ef`.
    const auto result_order = [](const ScoredNode &a, const ScoredNode &b) {
        if (a.orient_score != b.orient_score) {
            return a.orient_score < b.orient_score;
        }
        return a.id < b.id;
    };

    std::priority_queue<ScoredNode, std::vector<ScoredNode>,
                        decltype(candidate_order)>
        candidates(candidate_order);
    std::priority_queue<ScoredNode, std::vector<ScoredNode>,
                        decltype(result_order)>
        results(result_order);

    std::unordered_set<std::uint64_t> visited;
    visited.reserve(entry_points.size() * 2);

    for (const std::uint64_t entry : entry_points) {
        if (visited.insert(entry).second) {
            const ScoredNode node{entry, orient_of(entry)};
            candidates.push(node);
            results.push(node);
        }
    }

    while (!candidates.empty()) {
        const ScoredNode current = candidates.top();
        candidates.pop();

        if (results.size() >= ef && current.orient_score > results.top().orient_score) {
            break;
        }

        if (layer >= adjacency[current.id].size()) {
            continue;
        }

        for (const std::uint64_t neighbor : adjacency[current.id][layer]) {
            if (!visited.insert(neighbor).second) {
                continue;
            }

            const ScoredNode candidate_node{neighbor, orient_of(neighbor)};
            if (results.size() < ef ||
                candidate_node.orient_score < results.top().orient_score) {
                candidates.push(candidate_node);
                results.push(candidate_node);
                if (results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    std::vector<ScoredNode> output;
    output.reserve(results.size());
    while (!results.empty()) {
        output.push_back(results.top());
        results.pop();
    }
    std::reverse(output.begin(), output.end());
    return output;
}

std::vector<std::uint64_t> HnswIndex::select_neighbors_heuristic(
    std::uint64_t base_id, const std::vector<std::uint64_t> &candidates,
    std::size_t m) const {
    const bool prefer_higher = index_detail::higher_is_better(metric_);
    const float *base_vector = vectors_.get(base_id);

    struct Scored {
        std::uint64_t id;
        float orient_score;
    };

    std::vector<Scored> scored;
    scored.reserve(candidates.size());
    for (const std::uint64_t id : candidates) {
        if (id == base_id) {
            continue;
        }
        const float raw = index_detail::score_vector(
            metric_, base_vector, vectors_.get(id), vectors_.dim());
        scored.push_back({id, prefer_higher ? -raw : raw});
    }

    std::sort(scored.begin(), scored.end(),
             [](const Scored &a, const Scored &b) {
                 if (a.orient_score != b.orient_score) {
                     return a.orient_score < b.orient_score;
                 }
                 return a.id < b.id;
             });

    std::vector<std::uint64_t> selected;
    selected.reserve(std::min(m, scored.size()));

    for (const Scored &candidate : scored) {
        if (selected.size() >= m) {
            break;
        }

        bool keep = true;
        for (const std::uint64_t already_selected : selected) {
            const float raw_to_selected = index_detail::score_vector(
                metric_, vectors_.get(candidate.id),
                vectors_.get(already_selected), vectors_.dim());
            const float orient_to_selected =
                prefer_higher ? -raw_to_selected : raw_to_selected;
            if (orient_to_selected < candidate.orient_score) {
                keep = false;
                break;
            }
        }

        if (keep) {
            selected.push_back(candidate.id);
        }
    }

    return selected;
}

HnswIndex::InsertPlan HnswIndex::prepare_insert(
    const std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
    std::uint64_t entry_point, std::size_t max_level, bool has_entry_point,
    std::mt19937_64 &rng, std::uint64_t internal_id) const {
    const bool prefer_higher = index_detail::higher_is_better(metric_);
    const std::span<const float> query(vectors_.get(internal_id),
                                       vectors_.dim());
    const std::size_t level =
        random_level(rng, level_scale(config_.M));

    InsertPlan plan;
    plan.level = level;
    plan.own_neighbors.resize(level + 1);
    plan.becomes_entry_point = !has_entry_point || level > max_level;

    if (!has_entry_point) {
        return plan;
    }

    std::uint64_t current_entry = entry_point;
    for (std::size_t layer = max_level; layer > level; --layer) {
        const auto found = search_layer(query, {current_entry}, 1, layer,
                                        adjacency, prefer_higher);
        if (!found.empty()) {
            current_entry = found.front().id;
        }
    }

    std::vector<std::uint64_t> entry_set{current_entry};
    const std::size_t top_search_layer = std::min(level, max_level);

    for (std::size_t layer = top_search_layer + 1; layer-- > 0;) {
        const auto found = search_layer(query, entry_set,
                                        config_.ef_construction, layer,
                                        adjacency, prefer_higher);

        std::vector<std::uint64_t> candidate_ids;
        candidate_ids.reserve(found.size());
        for (const auto &node : found) {
            candidate_ids.push_back(node.id);
        }

        const std::size_t cap = (layer == 0) ? 2 * config_.M : config_.M;
        std::vector<std::uint64_t> chosen =
            select_neighbors_heuristic(internal_id, candidate_ids, cap);
        plan.own_neighbors[layer] = chosen;

        for (const std::uint64_t neighbor_id : chosen) {
            std::vector<std::uint64_t> updated =
                adjacency[neighbor_id][layer];
            updated.push_back(internal_id);
            if (updated.size() > cap) {
                updated =
                    select_neighbors_heuristic(neighbor_id, updated, cap);
            }
            plan.neighbor_updates.push_back(
                {neighbor_id, layer, std::move(updated)});
        }

        entry_set = std::move(candidate_ids);
    }

    return plan;
}

void HnswIndex::apply_insert(
    std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
    std::vector<std::size_t> &node_level, std::uint64_t &entry_point,
    std::size_t &max_level, bool &has_entry_point, std::uint64_t internal_id,
    InsertPlan plan) {
    adjacency.push_back(std::move(plan.own_neighbors));
    node_level.push_back(plan.level);

    for (auto &update : plan.neighbor_updates) {
        adjacency[update.node_id][update.layer] = std::move(update.neighbors);
    }

    if (plan.becomes_entry_point) {
        entry_point = internal_id;
        max_level = plan.level;
        has_entry_point = true;
    }
}

void HnswIndex::build() {
    std::vector<std::vector<std::vector<std::uint64_t>>> new_adjacency;
    std::vector<std::size_t> new_node_level;
    std::uint64_t new_entry_point = 0;
    std::size_t new_max_level = 0;
    bool new_has_entry_point = false;
    std::mt19937_64 new_rng(config_.seed);

    new_adjacency.reserve(vectors_.size());
    new_node_level.reserve(vectors_.size());

    for (std::uint64_t internal_id = 0; internal_id < vectors_.size();
         ++internal_id) {
        InsertPlan plan =
            prepare_insert(new_adjacency, new_entry_point, new_max_level,
                          new_has_entry_point, new_rng, internal_id);
        apply_insert(new_adjacency, new_node_level, new_entry_point,
                    new_max_level, new_has_entry_point, internal_id,
                    std::move(plan));
    }

    adjacency_ = std::move(new_adjacency);
    node_level_ = std::move(new_node_level);
    entry_point_ = new_entry_point;
    max_level_ = new_max_level;
    has_entry_point_ = new_has_entry_point;
    rng_ = new_rng;
    indexed_vector_count_ = vectors_.size();
    is_built_ = true;
}

void HnswIndex::add(std::uint64_t internal_id) {
    if (!is_built_) {
        throw std::logic_error("HNSW index must be built before insertion");
    }
    if (internal_id != indexed_vector_count_) {
        throw std::logic_error(
            "HNSW vectors must be added once in internal ID order");
    }

    InsertPlan plan = prepare_insert(adjacency_, entry_point_, max_level_,
                                     has_entry_point_, rng_, internal_id);

    adjacency_.reserve(adjacency_.size() + 1);
    node_level_.reserve(node_level_.size() + 1);

    apply_insert(adjacency_, node_level_, entry_point_, max_level_,
                has_entry_point_, internal_id, std::move(plan));
    ++indexed_vector_count_;
}

std::vector<InternalSearchResult> HnswIndex::search(
    std::span<const float> query, std::size_t top_k) const {
    if (query.size() != vectors_.dim()) {
        throw std::invalid_argument(
            "Query dimension does not match index dimension");
    }
    if (!is_built_) {
        throw std::logic_error("HNSW index must be built before searching");
    }
    if (indexed_vector_count_ != vectors_.size()) {
        throw std::logic_error(
            "HNSW index is stale and must be rebuilt before searching");
    }
    if (top_k == 0 || vectors_.size() == 0) {
        return {};
    }

    const bool prefer_higher = index_detail::higher_is_better(metric_);

    std::uint64_t current_entry = entry_point_;
    for (std::size_t layer = max_level_; layer > 0; --layer) {
        const auto found = search_layer(query, {current_entry}, 1, layer,
                                        adjacency_, prefer_higher);
        if (!found.empty()) {
            current_entry = found.front().id;
        }
    }

    const std::size_t ef = std::max(config_.ef_search, top_k);
    const auto found = search_layer(query, {current_entry}, ef, 0,
                                    adjacency_, prefer_higher);

    index_detail::TopKAccumulator results(std::min(top_k, found.size()),
                                          metric_);
    for (const auto &node : found) {
        const float raw_score = index_detail::score_vector(
            metric_, query.data(), vectors_.get(node.id), vectors_.dim());
        results.consider({node.id, raw_score});
    }

    return results.finish();
}

bool HnswIndex::is_built() const { return is_built_; }

const HnswConfig &HnswIndex::config() const { return config_; }

}  // namespace vectordb
