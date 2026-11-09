#include "../gpu/vk/bc_parts.h"

#include <cstdio>
#include <vector>

#define CHECK(name, condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %s at line %d\n", name, __LINE__); return 1; } } while (0)

using namespace mkwvk::bc_parts;

// The upload as vk_texture.cpp's upload_astc_gpu and texture_bc_continue run it: each recording takes runs of layers while its
// share lasts (at least one each call), the rest goes to the next recording. Returns the recordings it took, and checks every
// layer went once and in order, and no recording went past its share by more than the one run each call may always take.
int simulate(const std::vector<uint32_t>& all, uint64_t layer_texels, uint64_t budget, size_t max_run, int* recordings,
             size_t* staged = nullptr) {
    std::vector<uint32_t> pending = all, done;
    *recordings = 0;
    if (staged) *staged = 0;
    while (!pending.empty()) {
        ++*recordings;
        // (only what this recording takes is staged: fit_layers, before the copy into staging)
        const size_t take = fit_layers(pending.data(), pending.size(), max_run, layer_texels, 0, budget);
        if (take == 0) return 1;   // (every recording makes progress)
        if (staged) *staged += take;
        uint64_t used = 0;
        for (size_t r0 = 0; r0 < take;) {
            const size_t r1 = run_end(pending.data(), take, r0, max_run);
            const uint64_t run = layer_texels * (r1 - r0);
            if (r0 > 0 && used + run > budget && budget) return 1;
            used += run;
            for (size_t k = r0; k < r1; k++) done.push_back(pending[k]);
            r0 = r1;
        }
        pending.erase(pending.begin(), pending.begin() + (long)take);
        if (*recordings > 10000) return 1;
    }
    return done == all ? 0 : 1;
}

int main() {
    const uint32_t a[] = {0, 1, 2, 3, 7, 8, 9, 20};
    CHECK("a run ends at a gap", run_end(a, 8, 0, 8) == 4);
    CHECK("a run ends at the scratch's layers", run_end(a, 8, 0, 2) == 2);
    CHECK("a run after a gap", run_end(a, 8, 4, 8) == 7);
    CHECK("the last layer alone", run_end(a, 8, 7, 8) == 8);
    CHECK("no budget: everything", run_fits(1u << 30, 1u << 30, 0, false));
    CHECK("the first run always goes", run_fits(100, 1000, 10, true));
    CHECK("the others while they fit", run_fits(10, 6, 16, false) && !run_fits(11, 6, 16, false));

    // (Hyrule Field's terrain array: 90 layers of 1365 x 1365 decoded and 1536 x 1536 encoded, 8 levels: ~5.6 M texels a layer;
    // 8 layers a run, 16 M texels a recording)
    std::vector<uint32_t> terrain;
    for (uint32_t l = 0; l < 90; l++) terrain.push_back(l);
    const uint64_t layer = 5600000;
    int recs = 0;
    CHECK("the terrain array, in parts", simulate(terrain, layer, 16ull << 20, 8, &recs) == 0 && recs == 12);
    CHECK("all at once without a budget", simulate(terrain, layer, 0, 8, &recs) == 0 && recs == 1);
    CHECK("a big budget takes several runs a recording", simulate(terrain, layer, 256ull << 20, 8, &recs) == 0 && recs == 3);
    std::vector<uint32_t> dirty = {3, 4, 5, 40, 41, 89};
    CHECK("dirty layers only", simulate(dirty, layer, 16ull << 20, 8, &recs) == 0 && recs == 3);
    // (each layer is staged once over all the recordings: before, every continuation staged every layer still to come,
    // ~45 times the array on average for the terrain's 90 layers at 2 a recording)
    size_t staged = 0;
    CHECK("the terrain array staged once", simulate(terrain, layer, 16ull << 20, 1, &recs, &staged) == 0 && staged == 90 && recs > 1);
    CHECK("dirty layers staged once", simulate(dirty, layer, 16ull << 20, 1, &recs, &staged) == 0 && staged == dirty.size());
    CHECK("a recording with its share used takes one run", fit_layers(terrain.data(), 90, 8, layer, 16ull << 20, 16ull << 20) == 8);
    CHECK("fit_layers stops at a run that does not fit", fit_layers(terrain.data(), 90, 1, layer, 0, 16ull << 20) == 2);
    CHECK("no budget: all of them", fit_layers(terrain.data(), 90, 1, layer, 1ull << 40, 0) == 90);
    // (a new image's deferred layers: filled from the last layer the upload encoded, never from one still to come)
    CHECK("the fill comes from the last layer encoded", fill_source(terrain.data(), 3) == 2);
    CHECK("dirty layers: the last one encoded, not r0 - 1", fill_source(dirty.data(), 3) == 5);
    CHECK("nothing encoded: nothing to fill from", fill_source(terrain.data(), 0) == ~0u);
    // (a continuation that cannot record goes back a few times, then is given up: never dropped at the first failure)
    int tries = 0;
    while (retry_continue(tries)) tries++;
    CHECK("a failed continuation is retried", retry_continue(0) && tries == kContinueTries && tries > 1);
    std::puts("bc parts: all checks passed");
}
