#include "deepseek_v4_model.h"
#include "gemma4_model.h"
#include "glm_dsa_model.h"
#include "cuda_model_config.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto deepseek = mfq::cuda::deepseek_v4::Config::from_json(
            R"({"compress_ratios":[2,4,8],"num_hash_layers":1,"compress_rope_theta":160000,"rope_scaling":{"original_max_position_embeddings":4096,"factor":2,"beta_fast":16,"beta_slow":2}})",
            2);
        require(
            deepseek.compress_ratios == std::vector<std::int64_t>({2, 4}) &&
                deepseek.hash_layer_count == 1 &&
                deepseek.compress_rope_base == 160000.0 &&
                deepseek.rope_original_positions == 4096 &&
                deepseek.rope_factor == 2.0 &&
                deepseek.rope_beta_fast == 16.0 &&
                deepseek.rope_beta_slow == 2.0,
            "DeepSeek V4 typed config was not normalized");

        const auto gemma = mfq::cuda::gemma4::Config::from_json(
            R"({"global_head_dim":256,"num_global_key_value_heads":4,"sliding_window":1024,"attention_k_eq_v":true,"sliding_attention":{"rope_theta":5000}})",
            128, 2);
        require(
            gemma.global_head_dim == 256 &&
                gemma.num_global_key_value_heads == 4 &&
                gemma.sliding_window == 1024 &&
                gemma.sliding_rope_base == 5000.0 &&
                gemma.attention_key_equals_value,
            "Gemma4 typed config was not normalized");

        CudaRuntimeParameters common;
        common.num_hidden_layers = 3;
        const auto glm = mfq::cuda::glm_dsa::Config::from_json(
            R"({"indexer_types":["full","shared","full"],"mlp_layer_types":["dense","sparse","dense"]})",
            common);
        require(
            glm.indexer_types.size() == 3 &&
                glm.indexer_types[1] == "shared" &&
                glm.mlp_layer_types[1] == "sparse",
            "GLM DSA typed config was not normalized");

        bool rejected = false;
        try {
            static_cast<void>(mfq::cuda::glm_dsa::Config::from_json(
                R"({"indexer_types":["shared","full","full"],"mlp_layer_types":["dense","dense","dense"]})",
                common));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "invalid GLM DSA indexer schedule was accepted");

        std::cout << "CUDA typed config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA typed config tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
