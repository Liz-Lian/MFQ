"""校验桌面集成与前端源码契约；业务交互由 Studio 的行为测试补充覆盖。"""

import json
from pathlib import Path
from tests.studio_sources import read_studio_sources, read_studio_styles

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
APP_ENTRY = (STUDIO / "src" / "App.tsx").read_text(encoding="utf-8")
# 保持入口在前，允许组件与工具迁移到业务目录，避免把测试源码算作实现。
APP = APP_ENTRY + "\n" + "\n".join(
    path.read_text(encoding="utf-8")
    for directory in ("app", "features")
    for path in sorted((STUDIO / "src" / directory).rglob("*.ts*"))
    if ".test." not in path.name and ".spec." not in path.name
)
NAVIGATION = (STUDIO / "src" / "navigation.ts").read_text(encoding="utf-8")
API = (STUDIO / "src" / "api.ts").read_text(encoding="utf-8")
API += "\n" + "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((STUDIO / "src" / "shared" / "api").rglob("*.ts"))
    if ".test." not in path.name
)
MAIN = (STUDIO / "src" / "main.tsx").read_text(encoding="utf-8")
MARKDOWN = (STUDIO / "src" / "Markdown.tsx").read_text(encoding="utf-8")
MARKDOWN_TEXT = (STUDIO / "src" / "markdownText.ts").read_text(encoding="utf-8")
STUDIO_BRIDGE = (STUDIO / "src" / "studio.ts").read_text(encoding="utf-8")
PLATFORM_BRIDGE = STUDIO / "src" / "shared" / "platform" / "studio.ts"
if PLATFORM_BRIDGE.exists():
    STUDIO_BRIDGE += "\n" + PLATFORM_BRIDGE.read_text(encoding="utf-8")
STYLES = read_studio_styles()
REALTIME_AUDIO = (STUDIO / "src" / "realtimeAudio.ts").read_text(encoding="utf-8")
REALTIME_AUDIO += "\n" + "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((STUDIO / "src" / "features" / "voice").glob("*.ts"))
    if ".test." not in path.name
)
RELEASE_SCRIPT = (ROOT / "packaging" / "build_release_mac.sh").read_text(encoding="utf-8")


def test_studio_uses_one_package_for_web_and_desktop_clients():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    release_config = json.loads(
        (TAURI / "tauri.release-macos.conf.json").read_text(encoding="utf-8")
    )
    package = json.loads((STUDIO / "package.json").read_text(encoding="utf-8"))
    assert not (STUDIO / "web").exists()
    assert not (STUDIO / "desktop").exists()
    assert package["name"] == "@mfq/studio"
    assert package["scripts"]["tauri"] == "tauri"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["build"]["beforeBuildCommand"] == "npm run build"
    assert config["identifier"] == "com.tylogi.mfq-studio"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD
    assert "media-src 'self' asset: data: blob:" in config["app"]["security"]["csp"]
    assert release_config["bundle"]["macOS"]["hardenedRuntime"] is False
    assert 'if [[ "${mfq_signing_identity}" != "-" ]]' in RELEASE_SCRIPT
    assert '"hardenedRuntime":true' in RELEASE_SCRIPT
    assert "--remap-path-prefix=${HOME}=/mfq-build/home" in RELEASE_SCRIPT
    assert "--remap-path-prefix=${mfq_project_dir}=/mfq-src" in RELEASE_SCRIPT
    assert 'RUSTFLAGS="${mfq_release_rustflags}"' in RELEASE_SCRIPT
    assert "packaged Studio contains a private build path" in RELEASE_SCRIPT
    assert 'mfq-decode-metal" --self-test-metal' in RELEASE_SCRIPT


def test_assistant_markdown_recovers_fully_escaped_structural_line_breaks():
    assert "normalizeEscapedMarkdownLineBreaks" in MARKDOWN
    assert "normalizeEscapedLineBreaks={normalizeEscapedLineBreaks}" in APP
    assert 'message.role === "assistant"' in APP
    assert "isJsonDocument" in MARKDOWN_TEXT
    assert 'text.includes("```")' in MARKDOWN_TEXT
    assert "hasEscapedMarkdownStructure" in MARKDOWN_TEXT


def test_studio_starts_the_unified_local_server_and_bundled_runtime():
    assert 'Some("mfq-server")' in RUST
    assert "Command::new" in RUST
    assert "studio_start_local" in RUST
    assert '.arg("serve")' in RUST
    assert '.arg("--data-dir")' in RUST
    assert '.arg("--db")' not in RUST
    assert 'command.arg("--running-executable")' in RUST
    assert "MFQ_MLX_METALLIB" in RUST
    assert "MFQ_AVFOUNDATION_VIDEO_LIBRARY" in RUST


def test_studio_supports_local_and_remote_server_connections_with_voice_controls():
    models = read_studio_sources('features/models')
    chat = read_studio_sources('features/chat', 'features/voice')
    for command in ('studio_configure', 'studio_start_local', 'studio_select_model_directory'):
        assert command in RUST
    assert 'canUseNativeModelPicker' in models
    assert 'selectLocalModelDirectory' in models
    assert 'api.registerModelDirectory' in models
    assert 'ModelDirectoryDialog' in models
    assert 'jumpToModelDirectory' in models
    assert 'listing.current_path' in models
    assert 'RealtimeAudioController' in chat
    assert 'selectInteractionMode' in chat
    assert 'Browse folders on the MFQ Server host.' in models


def test_voice_component_prompt_requires_an_explicit_full_duplex_selection():
    chat = read_studio_sources('features/chat/ChatPage.tsx')
    toolbar = read_studio_sources('features/chat/components/ChatToolbar.tsx')
    assert "active?.mode === 'full_duplex'" in chat
    assert 'model_capabilities.features.audio_output' in chat
    assert '!inference.realtimeAvailable' in chat
    assert 'needsVoiceOutputComponent && voiceComponent' in chat
    assert 'features.audio_input' in toolbar
    assert 'features.full_duplex' in toolbar


def test_studio_handles_a_running_server_without_a_loaded_model():
    sessions = read_studio_sources('features/chat/hooks/useConversationSessions.ts')
    runtime = read_studio_sources('app/RuntimeProvider.tsx')
    assert 'if (!selectedModel || transitioning) return' in sessions
    assert 'modelAvailable' in sessions
    assert 'historyLoadedId === activeId' in sessions
    assert 'isRuntimeReady(status.runtime_state)' in runtime
    assert 'Promise.resolve<RuntimeModel[]>([])' in runtime
    assert 'No model loaded' in APP


def test_studio_exposes_every_loaded_model_and_switches_chat_sessions_safely():
    sessions = read_studio_sources('features/chat/hooks/useConversationSessions.ts')
    overview = read_studio_sources('features/runtime/OverviewPage.tsx')
    assert 'availableModelNames.map((name)' in overview
    assert 'artifacts.slice(0, 8)' not in APP
    assert '.forkSession(active.id, null, true, active.title, selectedModel)' in sessions
    assert 'setSelectedModel(session.model)' in sessions
    assert 'setActiveId(id)' in sessions
    assert 'active.model === selectedModel' in sessions
    assert 'generationBusy' in sessions
    assert 'controller.abort()' in sessions
    assert 'busy || sessionTransitioning' in APP
    assert 'model?: string' in API


def test_studio_uses_selected_runtime_mtp_availability():
    inference = read_studio_sources('features/chat/hooks/useChatInference.ts')
    assert 'instance?.mtp_supported' in inference
    assert 'instance?.mtp_available' in inference
    assert 'capabilities?.model === model' in inference
    assert 'enable_mtp: mtpSupported && mtpAvailable && effectiveSettings.enableMtp' in inference
    assert 'disabled={!mtpAvailable}' in APP
    assert 'mtp_supported?: boolean' in API
    assert 'mtp_available?: boolean' in API


def test_model_lifecycle_actions_stay_on_the_models_page():
    load_body = APP[
        APP.index("async function loadArtifact("):
        APP.index("async function finishModelRegistration(")
    ]
    unload_body = APP[
        APP.index("async function unloadInstance("):
        APP.index("const last = runtime?.last_request")
    ]
    assert 'navigate(STUDIO_PATHS.models)' in load_body
    assert 'navigate(STUDIO_PATHS.quantization)' not in load_body
    assert 'navigate(STUDIO_PATHS.models)' in unload_body
    assert 'navigate(STUDIO_PATHS.quantization)' not in unload_body


def test_model_hub_accepts_repository_links_and_downloads_into_the_model_catalog():
    assert "function parseHubReference(" in APP
    assert 'host === "huggingface.co"' in APP
    assert 'host === "modelscope.cn"' in APP
    assert 'destination: `models/${hubModel.provider}/${repositoryPath || name}`' in APP


def test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode():
    models = read_studio_sources('features/models')
    assert 'rfd::AsyncFileDialog::new()' in RUST
    assert '.pick_folder()' in RUST
    assert "tauri.invoke<string[] | null>('studio_select_model_directory')" in STUDIO_BRIDGE
    assert 'selectLocalModelDirectory()' in models
    assert 'finishModelRegistration(names)' in models
    assert 'Choose model folder' in models
    assert 'api.loadModel(' in models


def test_studio_uses_native_confirmation_dialogs_for_destructive_actions():
    assert "fn studio_confirm(message: String) -> bool" in RUST
    assert "rfd::MessageButtons::YesNo" in RUST
    assert "studio_confirm," in RUST
    assert "tauri.invoke<boolean>('studio_confirm', { message })" in STUDIO_BRIDGE
    assert "window.confirm" not in APP
    assert APP.count("await studioConfirm(") >= 6


def test_studio_has_a_render_error_boundary_instead_of_a_blank_window():
    assert "class AppErrorBoundary" in MAIN
    assert "static getDerivedStateFromError" in MAIN
    assert '<main className="fatal-error" role="alert">' in MAIN
    assert "<AppErrorBoundary>" in MAIN


def test_studio_loads_message_media_through_authenticated_blob_urls():
    assert "async fetchMedia(id: string, signal?: AbortSignal): Promise<Blob>" in API
    assert "headers: authorizedHeaders()" in API
    assert "api.fetchMedia(part.media.id, controller.signal)" in APP
    assert "URL.createObjectURL(blob)" in APP
    assert "URL.revokeObjectURL(objectUrl)" in APP
    assert 'alt="Attached image"' in APP
    assert "const src = api.mediaUrl(part.media.id)" not in APP


def test_studio_renders_video_first_frame_posters():
    assert "function VideoWithFirstFrame" in APP
    assert 'video.onloadeddata = () =>' in APP
    assert 'drawImage(video, 0, 0, canvas.width, canvas.height)' in APP
    assert 'canvas.toBlob((blob) =>' in APP
    assert 'poster={poster ?? undefined}' in APP
    assert '<VideoWithFirstFrame className="message-media media-video" controls src={src} />' in APP
    assert '<VideoWithFirstFrame muted src={attachment.previewUrl} />' in APP


def test_studio_validates_video_metadata_before_uploading_media():
    assert "const timeout = window.setTimeout(" in APP
    assert "video.load();" in APP
    metadata = APP.index("const metadata = await mediaMetadata(attachment.file, attachment.kind);")
    upload = APP.index("const resource = await api.uploadMedia(attachment.file);", metadata)
    assert metadata < upload


def test_studio_fetches_protected_message_media_and_documents():
    assert "const [loadFailed, setLoadFailed] = useState(false);" in APP
    assert "Unable to load attachment" in APP
    document = APP.index("async function downloadDocument()")
    assert "await api.fetchMedia(part.media.id)" in APP[document:]
    assert "anchor.download = part.name" in APP[document:]
    assert "document.body.appendChild(anchor)" in APP[document:]
    assert "anchor.remove()" in APP[document:]
    assert "href={api.mediaUrl(part.media.id)}" not in APP


def test_studio_drains_duplex_output_after_microphone_capture_stops():
    assert "MAX_RESPONSE_DRAIN_STEPS" in REALTIME_AUDIO
    assert "event.end_of_turn === true" in REALTIME_AUDIO
    assert "this.sendInput(new Float32Array(CHUNK_SAMPLES))" in REALTIME_AUDIO
    assert "this.callbacks.onText(target.buffer.sessionId, target.buffer.text)" in REALTIME_AUDIO


def test_studio_preserves_resampling_phase_across_audio_worklet_blocks():
    assert "class StreamingLinearResampler" in REALTIME_AUDIO
    assert "private position = 0" in REALTIME_AUDIO
    # 相位使用整数采样率累积，具体分块等价性由 audioCodec.test.ts 验证。
    assert "this.position += this.sourceRate" in REALTIME_AUDIO
    assert "this.position -= discard * this.targetRate" in REALTIME_AUDIO
    assert "new AudioContext({ sampleRate: INPUT_RATE })" in REALTIME_AUDIO
    assert "Math.round((input.length * targetRate) / sourceRate)" not in REALTIME_AUDIO


def test_studio_uses_the_model_bound_duplex_system_prompt():
    assert "modeTemplateSettings" in APP
    assert 'runtime?.duplex_sampling_defaults ?? realtime?.defaults' in APP
    assert "system_prompt: config.systemPrompt" in REALTIME_AUDIO
    assert "text_repetition_penalty: config.repetitionPenalty" in REALTIME_AUDIO
    assert "submitText(text, realtimeSessionConfig(active.id))" in APP


def test_studio_resolves_model_and_global_inference_settings_without_roles():
    inference = read_studio_sources('features/chat/hooks/useChatInference.ts')
    domain = read_studio_sources('features/chat/ChatProvider.tsx')
    assert 'inheritModelDefaults: true' in APP
    assert 'const effectiveSettings = useMemo' in inference
    assert 'modeTemplateSettings(settings, mode, runtime, realtime)' in inference
    assert 'sampling: inference.sampling' in domain
    assert 'system_prompt: inference.effectiveSettings.systemPrompt.trim()' in domain
    assert 'systemPrompt: value.systemPrompt.trim()' in domain
    assert 'max_tokens: effectiveSettings.maxTokens' in inference
    assert 'roleGenerationSettings' not in APP
    assert 'LANGUAGE_CONSISTENCY_PROMPT' not in APP
    assert 'Before answering, identify the language' not in APP


def test_studio_defaults_global_settings_to_inherited_model_parameters():
    assert 'className="settings-page-inherited" disabled={settingsDraft.inheritModelDefaults}' in APP
    assert 'checked={settingsDraft.inheritModelDefaults}' in APP
    assert "inherit_global_settings: preset.inheritGlobalSettings" in APP
    assert 'typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true' in APP
    assert 'typeof preset.metadata?.inherit_global_settings === "boolean"' in APP
    assert ".settings-page-inherited:disabled" in STYLES
    assert 'className="role-inherited-fields"' not in APP


def test_studio_exposes_theme_selection_without_using_sidebar_status_space():
    assert 'settingsDraft.theme' in APP
    assert '<option value="system">{tr("跟随系统", "System")}</option>' in APP
    assert '<option value="light">{tr("浅色", "Light")}</option>' in APP
    assert '<option value="dark">{tr("深色", "Dark")}</option>' in APP
    assert 'className="theme-switcher"' not in APP
    assert "connection-card" not in APP
    assert ".connection-card" not in STYLES


def test_studio_exposes_omlx_style_runtime_lifecycle_controls():
    models = read_studio_sources('features/models')
    assert 'className="runtime-hero"' in APP
    assert 'Pin in memory' in models
    assert 'Idle unload' in models
    assert 'pin: loadPinned' in models
    assert 'idle_ttl_seconds: loadIdleTtl' in models
    assert 'idle_ttl_seconds?: number | null' in API
    assert 'pin?: boolean' in API
    assert '.runtime-hero {' in STYLES


def test_studio_runtime_monogram_tracks_the_real_model_lifecycle():
    hero = read_studio_sources('features/runtime/RuntimeHero.tsx')
    assert "runtime?.model || 'Empty'" in hero
    assert "job.kind === 'model.load'" in hero
    assert "instance.state === 'loading'" in hero
    assert "instance.state === 'failed'" in hero
    assert 'name={modelHero.name} state={modelHero.state}' in hero
    assert 'runtime-status-pill ${modelHero.state}' in hero
    for state in ('loading', 'ready', 'failed'):
        assert f'.model-monogram.{state}' in STYLES


def test_studio_overview_lists_every_loaded_model():
    overview = read_studio_sources('features/runtime/OverviewPage.tsx')
    assert 'className="overview-models-panel"' in overview
    assert 'availableModelNames.map((name)' in overview
    assert "candidate.model === name && candidate.state !== 'failed'" in overview
    assert 'onClick={() => selectModel(name)}' in overview
    assert '.overview-model-grid {' in STYLES


def test_studio_adapts_prefix_cache_panel_to_flash_next_hot_cache():
    cache = read_studio_sources('features/runtime/CachePage.tsx')
    for field in ('prefix_cache_mode', 'prefix_cache_pending_bytes', 'prefix_cache_pending_max_bytes'):
        assert field in API
    assert 'single_device_hot_prefix' in cache
    assert 'Device-hot prefix' in cache
    assert 'Process lifetime' in cache
    assert 'api.clearRuntimeCache' in cache


def test_studio_uses_theme_aware_model_actions_and_readable_errors():
    assert ".panel-heading-actions button {" in STYLES
    assert "border: 1px solid var(--accent-border)" in STYLES
    assert ".mcp-form button { min-width: 64px;" in STYLES
    assert ".job-actions .secondary { border: 1px solid var(--panel-line);" in STYLES
    assert ".runtime-log p { min-width: 0; overflow-wrap: anywhere;" in STYLES
    assert ".error-banner span { min-width: 0; overflow-wrap: anywhere;" in STYLES


def test_realtime_turns_remain_bound_to_the_session_that_created_them():
    assert "sessionId: string;" in REALTIME_AUDIO
    assert "private clientSessionId: string | null = null" in REALTIME_AUDIO
    assert "sessionId: buffer.sessionId" in REALTIME_AUDIO
    assert "onTurn: ({ id, sessionId, text, audio })" in APP
    # 语音事件必须绑定事件携带的会话；文本生成可以独立保护当前选中会话。
    voice_callbacks = APP[APP.index('onInputStart:'):APP.index('// The controller reads')]
    assert "activeIdRef" not in voice_callbacks


def test_full_duplex_user_speech_visually_splits_assistant_turns():
    assert "SPEECH_RMS_THRESHOLD" in REALTIME_AUDIO
    assert "this.finishTurn();\n    this.inputTurnId = crypto.randomUUID()" in REALTIME_AUDIO
    assert "this.callbacks.onInputStart" in REALTIME_AUDIO
    assert "this.callbacks.onInputEnd" in REALTIME_AUDIO
    assert 'role: "user"' in APP
    assert "message.pending" in APP


def test_full_duplex_routes_pre_interrupt_response_tails_back_to_the_old_turn():
    assert "private pendingResponseTurns" in REALTIME_AUDIO
    assert "private responseTurnIds" in REALTIME_AUDIO
    assert "private currentInputTurnId" in REALTIME_AUDIO
    assert "private responseMessageIds" in REALTIME_AUDIO
    assert "private completedTurns" in REALTIME_AUDIO
    assert "this.lastCompletedByInputTurn.get(inputTurnId)" in REALTIME_AUDIO
    assert "this.publishTurn(target.buffer)" in REALTIME_AUDIO
    assert "message.id === id ? { ...message, text }" in APP


def test_closing_a_full_duplex_microphone_stops_instead_of_forcing_speech():
    assert "finishFullDuplexInput" not in REALTIME_AUDIO
    assert "} else if (this.audio.capturing) {\n      await this.stop();" in REALTIME_AUDIO
    assert "this.stopPlayback();" in REALTIME_AUDIO


def test_dashboard_uses_hivellm_style_static_backend_console_components():
    shell = read_studio_sources('app/StudioShell.tsx')
    for component in ('ScreenHeader', 'SectionLabel', 'TMPanel', 'MetricTile', 'SettingRow', 'UsageBar', 'EmptyPanel', 'PanelDeck'):
        assert f'function {component}' in APP
    assert 'className="overview-memory-panel"' in APP
    assert 'className="overview-footer-grid"' in APP
    assert 'PANEL_COLLAPSED_KEY' in APP
    assert 'aria-expanded={!isCollapsed}' in APP
    assert 'const storageKey = `${page}:${id}`' in APP
    assert 'localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(updated))' in APP
    assert '<Outlet />' in shell
    assert 'path="settings"' in APP_ENTRY
    assert 'path="quantization"' in APP_ENTRY
    assert 'api.clearCompletedJobs()' in APP
    assert 'api.deleteJob(id)' in APP
    assert 'className="drawer-scrim"' not in APP


def test_studio_streams_active_job_updates_without_polling_the_runtime():
    assert ".streamJobEvents(" in read_studio_sources('app/RuntimeProvider.tsx')
    assert "/api/v1/jobs/${id}/events/stream" in API
    assert "readEventStream(response, onEvent, signal)" in API
    assert "window.setInterval(() => void refreshRuntime(true), 2500)" not in APP


def test_server_settings_are_available_while_local_startup_is_pending():
    runtime = read_studio_sources('app/RuntimeProvider.tsx')
    shell = read_studio_sources('app/StudioShell.tsx')
    connection = read_studio_sources('features/connections/ConnectionsPage.tsx')
    assert runtime.index('setStudio(status)') < runtime.index('await startLocalStudio()')
    assert "location.pathname === '/runtime'" in shell
    assert 'await configureStudio(draft)' in connection
    assert 'if (credentialWritable) await saveStudioCredential(token)' in connection
    assert 'setCredentialWritable(true)' in connection
    assert 'await reloadService()' in connection


def test_server_page_matches_hivellm_information_architecture():
    connection = read_studio_sources('features/connections/ConnectionsPage.tsx', 'features/connections/MemorySettingsPanel.tsx', 'features/connections/InferenceDefaultsPanel.tsx')
    for label in ('Runtime', 'Memory plan', 'Persistent Prefix cache', 'Chat', 'Automation', 'Model ID', 'Bind address', 'Maximum output'):
        assert label in connection
    assert '<ToolsRoutingPanel />' in connection
    assert 'className="server-active-notice"' in connection
    assert 'path="runtime"' in APP_ENTRY
    assert '.server-active-notice' in STYLES
