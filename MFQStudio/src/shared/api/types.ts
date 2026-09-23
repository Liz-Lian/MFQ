/** 定义 MFQ HTTP 与实时协议的数据契约，不依赖 UI 或请求实现。 */
export type SessionMode = 'text' | 'voice' | 'full_duplex';
export type SessionState =
  | 'idle'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'interrupted'
  | 'reconnecting'
  | 'error'
  | 'closed';
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface MediaRef {
  id: string;
  sha256: string;
  mime_type: string;
  byte_size: number;
}

export interface MediaResource {
  media: MediaRef;
  created_at: string;
}

export interface DocumentResource {
  media: MediaRef;
  name: string;
  text: string;
  page_count?: number | null;
  extractor: string;
  created_at: string;
}

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'transcript'; text: string; language?: string | null }
  | { type: 'document'; media: MediaRef; name: string }
  | { type: 'image'; media: MediaRef; width?: number | null; height?: number | null }
  | {
      type: 'video';
      media: MediaRef;
      width?: number | null;
      height?: number | null;
      duration_ms?: number | null;
    }
  | {
      type: 'audio' | 'generated_audio';
      media: MediaRef;
      sample_rate_hz: number;
      channels: number;
      duration_ms?: number | null;
    }
  | { type: 'tool_call'; call_id: string; name: string; arguments: Record<string, unknown> }
  | { type: 'tool_result'; call_id: string; result: unknown; is_error: boolean };

export interface Session {
  id: string;
  model: string;
  mode: SessionMode;
  state: SessionState;
  revision: number;
  title: string | null;
  runtime_instance_id: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface SessionArchive {
  format: 'mfq-session-v1';
  session: Session;
  messages: Array<{ role: MessageRole; parts: ContentPart[]; created_at: string }>;
  media: Array<{
    sha256: string;
    mime_type: string;
    data_base64: string;
    document?: Record<string, unknown> | null;
  }>;
}

export interface Message {
  id: string;
  role: MessageRole;
  parts: ContentPart[];
  parent_id: string | null;
  created_at: string;
}

export interface SamplingParams {
  max_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  presence_penalty: number;
  frequency_penalty: number;
  repetition_penalty: number;
  seed?: number | null;
  enable_thinking: boolean;
  enable_vision: boolean;
  enable_mtp: boolean;
  mtp_max_draft_tokens?: number;
  reasoning_effort?: string | null;
}

export interface ResponsePerformance {
  prefill_tokens: number;
  ttft_ms: number;
  prefill_ms: number;
  prefill_tps: number;
  multimodal_ms: number;
  model_prefill_ms: number;
  processor_ms: number;
  complete_prefill_ms: number;
  complete_prefill_tps: number;
  decode_ms: number;
  decode_tps: number;
  generation_ms: number;
  complete_generation_ms: number;
  generation_tps: number;
  mtp_available?: boolean;
  mtp_used?: boolean;
  mtp_cycles?: number;
  mtp_drafted_tokens?: number;
  mtp_accepted_tokens?: number;
  mtp_acceptance_rate?: number;
  mtp_selected_depth?: number;
  mtp_depth_0_cycles?: number;
  mtp_depth_1_cycles?: number;
  mtp_depth_2_cycles?: number;
  mtp_depth_3_cycles?: number;
  mtp_depth_4_cycles?: number;
  mtp_depth_5_cycles?: number;
  mtp_position_1_acceptance_rate?: number;
  mtp_position_2_acceptance_rate?: number;
  mtp_position_3_acceptance_rate?: number;
  mtp_position_4_acceptance_rate?: number;
  mtp_position_5_acceptance_rate?: number;
  mtp_depth_0_cycle_ms?: number;
  mtp_depth_1_cycle_ms?: number;
  mtp_depth_2_cycle_ms?: number;
  mtp_depth_3_cycle_ms?: number;
  mtp_depth_4_cycle_ms?: number;
  mtp_depth_5_cycle_ms?: number;
  mtp_target_ms?: number;
  mtp_head_ms?: number;
  mtp_rollback_ms?: number;
  sampling: SamplingParams;
}

export interface ResponseResource {
  id: string;
  request_id: string;
  session_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  output_message_id?: string | null;
  output: ContentPart[];
  finish_reason?: string | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
  performance?: ResponsePerformance | null;
  settings?: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
  } | null;
  created_at: string;
  completed_at?: string | null;
}

export interface GenerationPresetResource {
  id: string;
  name: string;
  model?: string | null;
  mode?: SessionMode | null;
  settings: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
    input_role: 'user' | 'tool';
    tools: Array<Record<string, unknown>>;
    tool_choice: 'auto' | 'none' | 'required' | Record<string, unknown>;
    response_format: { type: 'text' | 'json_object' | 'json_schema'; [key: string]: unknown };
  };
  context_size: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ModelFeatureSet {
  text: boolean;
  image_input: boolean;
  video_input: boolean;
  audio_input: boolean;
  audio_output: boolean;
  full_duplex: boolean;
  mtp: boolean;
}

export interface ModelCapabilities {
  architecture_family: string;
  source: string;
  features: ModelFeatureSet;
}

export interface RuntimeCapabilities {
  model: string;
  model_type: string;
  model_capabilities: ModelCapabilities;
  vision_available: boolean;
  mtp_available: boolean;
  duplex_available: boolean;
}

export interface RuntimeRequestMetrics {
  id?: string;
  endpoint?: string;
  stream?: boolean;
  prompt_tokens?: number;
  prefill_tokens?: number;
  completion_tokens?: number;
  prefill_tps?: number;
  prefill_ms?: number;
  multimodal_ms?: number;
  model_prefill_ms?: number;
  processor_ms?: number;
  complete_prefill_ms?: number;
  complete_prefill_tps?: number;
  decode_tps?: number;
  decode_ms?: number;
  ttft_ms?: number;
  generation_ms?: number;
  complete_generation_ms?: number;
  generation_tps?: number;
  finish_reason?: string;
  client_connected?: boolean;
  completed_at?: number;
}

export interface RuntimeStatus {
  instance_id?: string;
  runtime_state?: string;
  model?: string;
  model_type?: string;
  model_capabilities?: ModelCapabilities;
  duplex_available?: boolean;
  active_requests?: number;
  total_requests?: number;
  failed_requests?: number;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  uptime_seconds?: number;
  max_context?: number;
  context_capacity?: number;
  reloading?: boolean;
  process_resident_bytes?: number | null;
  runtime_memory_budget_bytes?: number | null;
  runtime_memory_effective_budget_bytes?: number | null;
  runtime_memory_budget_mode?: 'automatic' | 'explicit' | 'disabled';
  runtime_memory_committed_bytes?: number;
  runtime_memory_headroom_bytes?: number | null;
  runtime_memory_pressure_level?: 'disabled' | 'normal' | 'soft' | 'hard';
  runtime_memory_pressure_ratio?: number | null;
  runtime_memory_shared_cache_reclaims?: number;
  runtime_memory_shared_cache_released_bytes?: number;
  runtime_memory_shared_cache_reclaim_failures?: number;
  mlx_active_bytes?: number;
  mlx_cache_bytes?: number;
  mlx_peak_bytes?: number;
  cuda_allocated_bytes?: number;
  cuda_reserved_bytes?: number;
  device_free_bytes?: number;
  device_total_bytes?: number;
  prefix_cache_queries?: number;
  prefix_cache_hits?: number;
  prefix_cache_hit_tokens?: number;
  prefix_cache_sessions?: number;
  prefix_cache_snapshots?: number;
  prefix_cache_tokens?: number;
  prefix_cache_bytes?: number;
  prefix_cache_max_sessions?: number;
  prefix_cache_max_snapshots_per_session?: number;
  prefix_cache_max_bytes?: number;
  prefix_cache_disk_blocks?: number;
  prefix_cache_disk_bytes?: number;
  prefix_cache_disk_max_bytes?: number;
  prefix_cache_hot_blocks?: number;
  prefix_cache_hot_bytes?: number;
  prefix_cache_pending_writes?: number;
  prefix_cache_pending_bytes?: number;
  prefix_cache_pending_max_bytes?: number;
  prefix_cache_writes?: number;
  prefix_cache_deduplicated_writes?: number;
  prefix_cache_disk_hits?: number;
  prefix_cache_hot_hits?: number;
  prefix_cache_evictions?: number;
  prefix_cache_corrupt_blocks?: number;
  prefix_cache_mode?: string;
  sampling_defaults?: Partial<SamplingParams>;
  duplex_sampling_defaults?: {
    system_prompt?: string;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    text_repetition_penalty?: number;
    [key: string]: unknown;
  };
  tts_sampling_defaults?: {
    temperature?: number;
    repetition_penalty?: number;
    [key: string]: unknown;
  };
  chat_template_capabilities?: {
    thinking?: { supported?: boolean };
    reasoning_effort?: { supported?: boolean; values?: string[] };
  };
  last_request?: RuntimeRequestMetrics | null;
  [key: string]: unknown;
}

export interface RuntimeModel {
  id: string;
  object?: string;
  owned_by?: string;
}

export interface ModelArtifact {
  id: string;
  name: string;
  architecture: string;
  format: 'mfq' | 'hf';
  shard_count: number;
  total_bytes: number;
  tensor_count: number;
  record_count: number;
  dtypes: string[];
  complete: boolean;
  loadable: boolean;
  modified_at: string;
  error?: string | null;
}

export interface ModelDirectoryEntry {
  id: string;
  name: string;
  model_file_count: number;
}

export interface ModelDirectoryList {
  current_id: string | null;
  current_name: string | null;
  current_path: string | null;
  parent_id: string | null;
  model_file_count: number;
  data: ModelDirectoryEntry[];
}

export interface RuntimeInstance {
  id: string;
  model: string;
  state: 'loading' | 'ready' | 'busy' | 'unloading' | 'failed';
  devices: string[];
  active_sessions: number;
  queued_requests: number;
  resident_bytes?: number | null;
  kv_bytes?: number | null;
  context_size?: number | null;
  started_at?: string | null;
  last_used_at?: string | null;
  idle_ttl_seconds?: number | null;
  pinned?: boolean;
  mtp_supported?: boolean;
  mtp_available?: boolean;
  error?: ApiErrorBody['error'] | null;
}

export interface RuntimeProfile {
  id: string;
  name: string;
  load: {
    model: string;
    artifact_uri?: string | null;
    device_ids: string[];
    idle_ttl_seconds?: number | null;
    pin: boolean;
    context_size: number;
    prefill_chunk_size: number;
    moe_gpu_cache_gb?: number | null;
    prefix_cache_max_sessions?: number | null;
    prefix_cache_max_snapshots_per_session?: number | null;
    prefix_cache_max_bytes?: number | null;
    prefix_cache_enabled?: boolean;
    prefix_cache_disk_bytes?: number | null;
    prefix_cache_hot_bytes?: number | null;
    prefix_cache_block_tokens?: number | null;
    prefix_cache_pending_bytes?: number | null;
    sampling_defaults?: SamplingParams | null;
  };
  artifact_id: string;
  artifact_modified_at: string;
  drifted: boolean;
  drift_reason?: string | null;
  created_at: string;
  updated_at: string;
}

export interface HubModelSummary {
  provider: 'huggingface' | 'modelscope';
  repo_id: string;
  downloads: number;
  likes: number;
  total_bytes: number;
  updated_at?: string | null;
}

export interface HubModelInfo extends HubModelSummary {
  revision: string;
  files: Array<{ name: string; byte_size: number }>;
  tags: string[];
}

export interface ArtifactLineage {
  id: string;
  artifact_uri: string;
  artifact_name: string;
  producer_job_id: string;
  producer_kind: string;
  source_uris: string[];
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  validation_job_ids: string[];
  created_at: string;
}

export interface DatasetResource {
  id: string;
  name: string;
  kind: 'wikitext2' | 'custom';
  artifact_uri: string;
  sha256: string;
  byte_size: number;
  source_uri?: string | null;
  revision?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface EvaluationResult {
  id: string;
  job_id: string;
  kind: 'perplexity' | 'kernel_benchmark';
  model_id: string;
  metrics: Record<string, unknown>;
  parameters: Record<string, unknown>;
  dataset_id?: string | null;
  dataset_manifest: Record<string, unknown>;
  hardware_identity: Record<string, unknown>;
  runtime_identity: Record<string, unknown>;
  comparison_key: string;
  created_at: string;
}

export interface EvaluationComparison {
  comparison_key: string;
  baseline_id: string;
  metrics: string[];
  rows: Array<{
    evaluation: EvaluationResult;
    deltas: Record<string, number | null>;
    ratios: Record<string, number | null>;
  }>;
}

export interface RemoteNode {
  id: string;
  name: string;
  url: string;
  api_key_env?: string | null;
  enabled: boolean;
  healthy: boolean;
  models: string[];
  active_requests: number;
  metrics: Record<string, unknown>;
  last_checked_at?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RuntimeMetricSnapshot {
  sequence: number;
  instance_id?: string | null;
  model?: string | null;
  values: RuntimeStatus;
  captured_at: string;
}

export interface RuntimeLogEntry {
  sequence: number;
  instance_id?: string | null;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  fields: Record<string, unknown>;
  created_at: string;
}

export interface JobResource {
  id: string;
  kind: string;
  status:
    'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
  payload: Record<string, unknown>;
  progress: number;
  cancel_requested: boolean;
  result?: Record<string, unknown> | null;
  error?: ApiErrorBody['error'] | null;
  created_at: string;
  updated_at: string;
}

export interface JobEventResource {
  job_id: string;
  sequence: number;
  type: 'state' | 'progress' | 'log' | 'artifact';
  level: RuntimeLogEntry['level'];
  message?: string | null;
  progress?: number | null;
  data: Record<string, unknown>;
  created_at: string;
}

export interface JsonSchemaProperty {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  anyOf?: JsonSchemaProperty[];
  items?: JsonSchemaProperty;
}

export interface JobKindResource {
  kind: string;
  payload_schema: {
    type?: string;
    properties?: Record<string, JsonSchemaProperty>;
    required?: string[];
    [key: string]: unknown;
  };
}

export interface McpServerResource {
  id: string;
  name: string;
  transport: 'stdio' | 'streamable_http';
  enabled: boolean;
  url?: string | null;
  command?: string | null;
  args: string[];
  header_env: Record<string, string>;
  timeout_seconds: number;
  created_at: string;
  updated_at: string;
}

export interface McpToolResource {
  server_id: string;
  server: string;
  name: string;
  qualified_name: string;
  description?: string | null;
  input_schema: Record<string, unknown>;
}

export interface McpToolCallResult {
  server: string;
  name: string;
  content: Array<Record<string, unknown>>;
  structured_content?: Record<string, unknown> | null;
  is_error: boolean;
}

export interface RealtimeCapabilities {
  available: boolean;
  input?: string[];
  output?: string[];
  input_sample_rate?: number;
  output_sample_rate?: number;
  defaults?: Record<string, unknown>;
  model_capabilities?: ModelCapabilities;
}

export interface VoiceOutputComponentStatus {
  id: string;
  state: 'missing' | 'installing' | 'ready';
  ready: boolean;
  installed_bytes: number;
  total_bytes: number;
  repository: string;
  revision: string;
  active: boolean;
  supported_model_loaded: boolean;
  model?: string | null;
  error?: string | null;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details: Record<string, unknown>;
  };
}

export interface RealtimeFrame {
  protocol_version: '1.0';
  session_id: string;
  sequence: number;
  timestamp: string;
  payload: Record<string, unknown> & { type: string };
}

export interface StreamRequest {
  request_id: string;
  expected_revision: number;
  input: ContentPart[];
  input_role?: 'user' | 'tool';
  sampling: SamplingParams;
  system_prompt?: string | null;
  include_reasoning_history: boolean;
  tools?: Array<{
    type: 'function';
    function: {
      name: string;
      description?: string | null;
      parameters: Record<string, unknown>;
    };
  }>;
  tool_choice?: 'auto' | 'none' | 'required';
  stream: true;
}
