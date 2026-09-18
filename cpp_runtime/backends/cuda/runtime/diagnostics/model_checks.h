#pragma once

#include "../causal_lm.h"

#include <cstdint>
#include <string>
#include <vector>

struct KlReferenceContract {
    std::int64_t n_batch = 0;
    std::int64_t n_ubatch = 0;
};

enum class KlEvaluator {
    Legacy,
    Optimized,
};

KlEvaluator parse_kl_evaluator(const std::string& value);
KlMmqMode parse_kl_mmq_mode(const std::string& value);
std::vector<KlMmqMode> parse_kl_mmq_sequence(const std::string& value);
const char* kl_evaluator_name(KlEvaluator evaluator);
template <typename Model>
int run_kl_eval_batched(Model& model, const std::string& reference_path, int max_chunks, std::int64_t requested_n_batch, int score_override, const KlReferenceContract& reference_contract);
template <typename Model>
int run_selected_kl_eval(Model& model, const std::string& reference_path, int max_chunks, KlEvaluator evaluator, std::int64_t requested_n_batch, int score_override, const KlReferenceContract& reference_contract);
int run_kl_eval_streamed(const std::string& model_path, const std::string& config_path, const std::string& reference_path, const std::string& logits_output_path, int max_chunks, int layer_group, int chunk_batch, int score_override, const KlReferenceContract& reference_contract);
int run_linear_check(const std::string& model_path, const std::string& name, int rows, int gate_mode, int repetitions);
int run_cpu_linear_check(const std::string& model_path, const std::string& name, int rows, int gate_mode, int repetitions);
int run_tensor_parallel_linear_check(const std::string& model_path, const std::string& name, TensorParallelAxis axis, int rows);
std::vector<std::string> parse_tensor_names(const std::string& value);
int run_linear_group_check(const std::string& model_path, const std::string& names, int rows, int repetitions);
int run_gdn_operator_check(const std::string& input_dir, const std::string& output_path, const std::string& state_path, std::int64_t tokens, std::int64_t query_heads, std::int64_t value_heads, std::int64_t head_dim);
int run_linear_conv_operator_check(const std::string& input_dir, const std::string& output_dir, std::int64_t tokens, std::int64_t query_heads, std::int64_t value_heads, std::int64_t key_dim, std::int64_t value_dim, std::int64_t kernel_size, double eps);
int run_q8_embedding_check(const std::string& model_path, const std::string& name);
int run_dsv4_output_a_check(const std::string& model_path, const std::string& name, int batch, int repetitions);
int run_gemma_geglu_check(const std::string& model_path, int layer, int repetitions);
int run_expert_parallel_moe_check(const std::string& model_path, const std::string& tensor_name, int tokens, int routes);
int run_mfe_tensor_check(const std::string& model_path, const std::string& tensor_name, int tokens, int routes, int repetitions, int split_width, bool routed_input, bool benchmark_only);
int run_moe_check(const std::string& model_path, const std::string& config_path, int layer, const std::vector<std::int64_t>& token_sizes, int repetitions);
int run_attention_decode_check(int length, int repetitions, int dimension, bool sliding, int window);
int run_gemma4_swa_check(int repetitions);
int run_glm_dsa_check(int repetitions);
int run_dsv4_hc_check(int repetitions);
int run_dsv4_attention_check(int repetitions);
int run_text_session_state_check();
