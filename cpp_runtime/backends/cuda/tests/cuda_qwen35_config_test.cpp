#include "qwen35_model.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        mfq::ModelGraph graph;
        graph.backbone = "qwen3_5";
        const auto config = mfq::cuda::qwen35::Config::from_json(R"json({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 192,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "head_dim": 16,
                "attn_output_gate": true,
                "layer_types": ["full_attention", "linear_attention"],
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.5
                    }
                }
            }
        })json", graph);

        require(config.model_type == "qwen3_5", "model type was not parsed");
        require(config.hidden_size == 64 && config.head_dim == 16,
                "text geometry was not parsed");
        require(config.layer_types.size() == 2 &&
                    config.layer_types[1] == "linear_attention",
                "layer schedule was not parsed");
        require(config.attention_output_gate,
                "attention output gate was not parsed");
        require(
            config.linear_num_key_heads == 2 &&
                config.linear_num_value_heads == 4 &&
                config.linear_k_size() == 256 &&
                config.linear_v_size() == 512,
            "linear-attention geometry was not retained by typed config");
        require(config.rope_base == 10000.0 && config.rotary_dim == 8,
                "RoPE configuration was not parsed");
        require(!config.grid_vision,
                "text-only graph unexpectedly loaded vision config");

        std::cout << "CUDA Qwen3.5 config test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA Qwen3.5 config test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
