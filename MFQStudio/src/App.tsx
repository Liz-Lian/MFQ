/** MFQ Studio 应用编排：组合业务模块、会话操作与桌面运行时。 */
import { FormEvent, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { ApiError, ArtifactLineage, ContentPart, DatasetResource, EvaluationComparison, EvaluationResult, HubModelInfo, HubModelSummary, JobResource, JobKindResource, JsonSchemaProperty, McpServerResource, McpToolResource, Message, ModelArtifact, ModelDirectoryList, RealtimeCapabilities, RealtimeFrame, ResponseResource, RuntimeCapabilities, RuntimeInstance, RuntimeLogEntry, RuntimeModel, RuntimeProfile, RemoteNode, RuntimeRequestMetrics, RuntimeStatus, SamplingParams, SessionArchive, Session, SessionMode, VoiceOutputComponentStatus, api, setApiBaseUrl, setApiToken, streamResponse } from './api';
import { RealtimeAudioController, VoiceState, saveVoiceClip } from './realtimeAudio';
import { StudioConfig, StudioStatus, configureStudio, isStudio, saveStudioCredential, selectLocalModelDirectory, startLocalStudio, studioConfirm, studioCredential, studioStatus } from './studio';
import { DashboardPage, LabPage, STUDIO_PATHS, dashboardPath, labPath, normalizeStudioPath, resolveStudioLocation } from './navigation';
import { useUiStore } from './stores/uiStore';
import { UiLanguage, UiTheme, PresetName, GenerationSettings, SETTINGS_KEY, DEFAULT_SETTINGS, PRESETS, modeTemplateSettings, loadSettings } from './features/settings/configuration';
import { StoredPreset, STORED_PRESETS_KEY, presetSnapshot, loadStoredPresets, storedPresetFromResource, presetResourceBody } from './features/settings/presets';
import { parseHubReference } from './features/models/hubReference';
import { isRuntimeReady, runtimeModelNames, runtimeSelectionNames } from './features/runtime/modelSelection';
import { displayPrefillMetric, preferPositiveMetric } from './features/runtime/metrics';
import { schemaDefault, schemaType, isTerminalJob } from './features/jobs/jobSchema';
import { PendingAttachment, DOCUMENT_ACCEPT, MAX_DOCUMENT_BYTES, isTextDocument, mediaMetadata, documentMimeType } from './features/chat/attachments';
import { textParts, isMediaPart } from './features/chat/messageParts';
import { VideoWithFirstFrame, MediaPartView, DocumentPartView } from './features/chat/MessageMedia';
import { VoiceMessage, LiveVoiceOutput, VOICE_HISTORY_KEY, loadVoiceHistory } from './features/voice/history';
import { AudioClip } from './features/voice/AudioClip';
import { errorMessage, formatNumber, formatBytes, formatDuration } from './app/formatters';
import { Icon, ScreenHeader, SectionLabel, TMPanel, ModelMonogram, MetricTile, SettingRow, UsageBar, EmptyPanel } from './app/display';
import { PanelDeck } from './app/PanelDeck';
import { SettingsPage } from './features/settings/SettingsPage';
import { renderMarkdown } from './features/chat/MessageMarkdown';
import { SavedMessageList } from './features/chat/SavedMessageList';



interface LiveOutput {
  reasoning: string;
  text: string;
  tools: string[];
}










interface EditDraft {
  messageId: string;
  text: string;
}







const MODE_LABELS: Record<SessionMode, [string, string]> = {
  text: ["文本", "Text"],
  voice: ["语音", "Voice"],
  full_duplex: ["全双工", "Full duplex"],
};








































export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const { view, dashboardPage, labPage } = resolveStudioLocation(location.pathname);
  const normalizedPathname = normalizeStudioPath(location.pathname);
  useEffect(() => {
    const titles: Record<string, string> = {
      '/': 'MFQ Studio',
      '/chat': 'Chat · MFQ Studio',
      '/models': 'Models · MFQ Studio',
      '/runtime': 'Runtime · MFQ Studio',
      '/resources': 'Resources · MFQ Studio',
      '/model-hub': 'Model hub · MFQ Studio',
      '/evaluations': 'Evaluations · MFQ Studio',
      '/quantization': 'Quantization · MFQ Studio',
      '/logs': 'Logs · MFQ Studio',
      '/settings': 'Settings · MFQ Studio',
    };
    document.title = titles[normalizedPathname] ?? 'MFQ Studio';
  }, [normalizedPathname]);
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const closeSidebar = useUiStore((state) => state.closeSidebar);
  const openSidebar = useUiStore((state) => state.openSidebar);
  useEffect(() => {
    closeSidebar();
  }, [closeSidebar, location.pathname]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [responses, setResponses] = useState<Record<string, ResponseResource>>({});
  const [voiceMessages, setVoiceMessages] = useState<VoiceMessage[]>(loadVoiceHistory);
  const [models, setModels] = useState<RuntimeModel[]>([]);
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [instances, setInstances] = useState<RuntimeInstance[]>([]);
  const [runtimeProfiles, setRuntimeProfiles] = useState<RuntimeProfile[]>([]);
  const [remoteNodes, setRemoteNodes] = useState<RemoteNode[]>([]);
  const [nodeDraft, setNodeDraft] = useState({ name: "", url: "", api_key_env: "" });
  const [lineage, setLineage] = useState<ArtifactLineage[]>([]);
  const [datasets, setDatasets] = useState<DatasetResource[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [selectedEvaluations, setSelectedEvaluations] = useState<string[]>([]);
  const [evaluationComparison, setEvaluationComparison] = useState<EvaluationComparison | null>(null);
  const [datasetDraft, setDatasetDraft] = useState({ name: "", artifact_uri: "", kind: "custom" as DatasetResource["kind"] });
  const [jobs, setJobs] = useState<JobResource[]>([]);
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);
  const [mcpServers, setMcpServers] = useState<McpServerResource[]>([]);
  const [mcpTools, setMcpTools] = useState<McpToolResource[]>([]);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [mcpDraft, setMcpDraft] = useState({
    name: "",
    transport: "streamable_http" as "stdio" | "streamable_http",
    endpoint: "",
  });
  const [selectedJobKind, setSelectedJobKind] = useState("");
  const [jobPayload, setJobPayload] = useState<Record<string, unknown>>({});
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [imatrixImporting, setImatrixImporting] = useState(false);
  const [pendingImatrix, setPendingImatrix] = useState("");
  const imatrixInputRef = useRef<HTMLInputElement | null>(null);
  const [jobLogs, setJobLogs] = useState<RuntimeLogEntry[]>([]);
  const [hubProvider, setHubProvider] = useState<HubModelSummary["provider"]>("modelscope");
  const [hubQuery, setHubQuery] = useState("");
  const [hubResults, setHubResults] = useState<HubModelSummary[]>([]);
  const [hubModel, setHubModel] = useState<HubModelInfo | null>(null);
  const [runtimeLogs, setRuntimeLogs] = useState<RuntimeLogEntry[]>([]);
  const [model, setModel] = useState("");
  const [mode, setMode] = useState<SessionMode>("text");
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [live, setLive] = useState<LiveOutput | null>(null);
  const [busy, setBusy] = useState(false);
  const [sessionTransitioning, setSessionTransitioning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [modelFilter, setModelFilter] = useState('');
  const [chatSessionsOpen, setChatSessionsOpen] = useState(true);
  const [realtime, setRealtime] = useState<RealtimeCapabilities | null>(null);
  const [realtimeAvailable, setRealtimeAvailable] = useState(false);
  const [voiceComponent, setVoiceComponent] = useState<VoiceOutputComponentStatus | null>(null);
  const [voiceComponentBusy, setVoiceComponentBusy] = useState(false);
  const [requestHistory, setRequestHistory] = useState<RuntimeRequestMetrics[]>([]);
  const [settings, setSettings] = useState<GenerationSettings>(loadSettings);
  const [settingsDraft, setSettingsDraft] = useState<GenerationSettings>(settings);
  const [storedPresets, setStoredPresets] = useState<StoredPreset[]>(loadStoredPresets);
  const [selectedStoredPreset, setSelectedStoredPreset] = useState("");
  const [storedPresetName, setStoredPresetName] = useState("");
  const [presetStatus, setPresetStatus] = useState<{ error: boolean; text: string } | null>(null);
  const [contextSize, setContextSize] = useState(32768);
  const [loadPinned, setLoadPinned] = useState(false);
  const [loadIdleTtl, setLoadIdleTtl] = useState<number | null>(null);
  const [profileName, setProfileName] = useState("");
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [studioDraft, setStudioDraft] = useState<StudioConfig | null>(null);
  const [modelBrowser, setModelBrowser] = useState<ModelDirectoryList | null>(null);
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false);
  const [modelDirectoryPath, setModelDirectoryPath] = useState("");
  const [studioToken, setStudioToken] = useState("");
  const [studioCredentialWritable, setStudioCredentialWritable] = useState(false);
  const [endpointCopied, setEndpointCopied] = useState(false);
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [liveVoice, setLiveVoice] = useState<LiveVoiceOutput | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement | null>(null);
  const messageScrollerRef = useRef<HTMLDivElement | null>(null);
  const autoFollowOutputRef = useRef(true);
  const voiceRef = useRef<RealtimeAudioController | null>(null);
  const voiceClipWrites = useRef(new Map<string, Promise<void>>());
  const appliedModeTemplate = useRef("");
  const sessionSwitchRef = useRef("");
  const selectedRuntimeInstanceIdRef = useRef<string | null>(null);

  const english =
    settings.language === "en" ||
    (settings.language === "system" && !navigator.language.toLowerCase().startsWith("zh"));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const canUseNativeModelPicker = isStudio() && studio?.config.mode !== "remote";
  const panelLabels = useMemo(() => ({
    collapse: tr("收起面板", "Collapse panel"),
    expand: tr("展开面板", "Expand panel"),
  }), [tr]);
  const active = useMemo(
    () => sessions.find((session) => session.id === activeId) ?? null,
    [activeId, sessions],
  );
  const availableModelNames = useMemo(
    () => runtimeModelNames(models, instances),
    [instances, models],
  );
  const filteredInstances = useMemo(() => {
    const query = modelFilter.trim().toLowerCase();
    return instances.filter((instance) => !query || instance.model.toLowerCase().includes(query));
  }, [instances, modelFilter]);
  const filteredArtifacts = useMemo(() => {
    const query = modelFilter.trim().toLowerCase();
    return artifacts.filter((artifact) => !query || artifact.name.toLowerCase().includes(query));
  }, [artifacts, modelFilter]);
  const selectedModelAvailable = Boolean(model && availableModelNames.includes(model));
  const conversationReady = Boolean(
    active && selectedModelAvailable && active.model === model,
  );
  const selectedRuntimeInstance = instances.find(
    (instance) => instance.model === model && instance.state !== "failed",
  );
  selectedRuntimeInstanceIdRef.current = selectedRuntimeInstance?.id ?? null;
  const selectedModelLoading = selectedRuntimeInstance?.state === "loading" || jobs.some(
    (job) => job.kind === "model.load"
      && job.payload.model === model
      && ["queued", "running", "cancelling"].includes(job.status),
  );
  const activeJobIds = jobs
    .filter((job) => ["queued", "running", "cancelling"].includes(job.status))
    .map((job) => job.id)
    .sort()
    .join(",");
  const currentVoiceMessages = useMemo(
    () => voiceMessages.filter((message) => message.sessionId === activeId),
    [activeId, voiceMessages],
  );
  const resolvedGlobalSettings = useMemo(
    () => settings.inheritModelDefaults
      ? modeTemplateSettings(settings, active?.mode ?? mode, runtime, realtime)
      : settings,
    [active?.mode, mode, realtime, runtime, settings],
  );
  const effectiveSettings = resolvedGlobalSettings;
  const effectiveSystemPrompt = effectiveSettings.systemPrompt.trim();
  const reasoningValues = useMemo(() => {
    const values = runtime?.chat_template_capabilities?.reasoning_effort?.values;
    return Array.isArray(values) ? values : [];
  }, [runtime]);
  const thinkingSupported =
    runtime?.chat_template_capabilities?.thinking?.supported === true;
  const visionSupported = Boolean(
    capabilities?.model_capabilities.features.image_input
      || capabilities?.model_capabilities.features.video_input,
  );
  const visionAvailable = Boolean(
    capabilities?.vision_available || visionSupported,
  );
  const mtpSupported = typeof selectedRuntimeInstance?.mtp_supported === "boolean"
    ? selectedRuntimeInstance.mtp_supported
    : capabilities?.model === model
      && capabilities.model_capabilities.features.mtp === true;
  const mtpAvailable = typeof selectedRuntimeInstance?.mtp_available === "boolean"
    ? selectedRuntimeInstance.mtp_available
    : capabilities?.model === model && capabilities.mtp_available === true;
  const attachmentAccept = useMemo(() => {
    const features = capabilities?.model_capabilities.features;
    return [
      effectiveSettings.enableVision && features?.image_input ? "image/*" : "",
      effectiveSettings.enableVision && features?.video_input ? "video/*" : "",
      features?.audio_input ? "audio/*" : "",
      DOCUMENT_ACCEPT,
    ]
      .filter(Boolean)
      .join(",");
  }, [capabilities, effectiveSettings.enableVision]);

  const refreshSessions = useCallback(async (preferredId?: string) => {
    const next = await api.listSessions();
    setSessions(next);
    setActiveId((current) => {
      const wanted = preferredId ?? current;
      if (wanted && next.some((session) => session.id === wanted)) return wanted;
      return next[0]?.id ?? null;
    });
  }, []);

  const refreshMcp = useCallback(async () => {
    const [servers, tools] = await Promise.all([api.mcpServers(), api.mcpTools()]);
    setMcpServers(servers);
    setMcpTools(tools.data);
  }, []);

  const refreshRuntime = useCallback(async (quiet = true) => {
    try {
      const management = await Promise.all([
        api.runtimeMetrics(200),
        api.modelArtifacts(),
        api.runtimeInstances(),
        api.runtimeProfiles(),
        api.jobs(100),
        api.runtimeLogs(100),
        api.jobKinds(),
        api.artifactLineage(),
        api.datasets(),
        api.evaluations(),
        api.remoteNodes(),
      ]);
      const [metricHistory, nextArtifacts, nextInstances, nextProfiles, nextJobs, nextLogs, nextKinds, nextLineage, nextDatasets, nextEvaluations, nextNodes] = management;
      const requestedInstanceId = selectedRuntimeInstanceIdRef.current;
      const selectedInstanceId = nextInstances.some(
        (instance) => instance.id === requestedInstanceId,
      ) ? requestedInstanceId : null;
      const [statusResult, voiceComponentResult] = await Promise.allSettled([
        api.runtimeStatus(selectedInstanceId),
        api.voiceOutputComponent(),
      ]);
      const status = statusResult.status === "fulfilled" ? statusResult.value : null;
      let nextCapabilities: RuntimeCapabilities | null = null;
      let nextModels: RuntimeModel[] = [];
      let nextRealtime: RealtimeCapabilities | null = null;
      if (isRuntimeReady(status?.runtime_state)) {
        const [capabilityResult, modelResult, realtimeResult] = await Promise.allSettled([
          api.runtimeCapabilities(selectedInstanceId),
          api.runtimeModels(),
          api.realtimeCapabilities(),
        ]);
        if (capabilityResult.status === "fulfilled") nextCapabilities = capabilityResult.value;
        if (modelResult.status === "fulfilled") nextModels = modelResult.value;
        if (realtimeResult.status === "fulfilled") nextRealtime = realtimeResult.value;
      }
      setCapabilities(nextCapabilities);
      setModels(nextModels);
      setRuntime(status);
      setRealtime(nextRealtime);
      setRealtimeAvailable(nextRealtime?.available === true);
      const nextModelNames = runtimeModelNames(nextModels, nextInstances);
      const nextSelectionNames = runtimeSelectionNames(nextModels, nextInstances, nextJobs);
      const statusModel = typeof status?.model === "string" ? status.model : "";
      setModel((current) => {
        if (current && nextSelectionNames.includes(current)) return current;
        if (statusModel && nextModelNames.includes(statusModel)) return statusModel;
        return nextModelNames[0] ?? "";
      });
      if (voiceComponentResult.status === "fulfilled") {
        setVoiceComponent(voiceComponentResult.value);
      }
      setArtifacts(nextArtifacts);
      setInstances(nextInstances);
      setRuntimeProfiles(nextProfiles);
      setJobs(nextJobs);
      setRuntimeLogs(nextLogs);
      setJobKinds(nextKinds);
      setLineage(nextLineage);
      setDatasets(nextDatasets);
      setEvaluations(nextEvaluations);
      setRemoteNodes(nextNodes);
      setSelectedJobKind((current) => current || nextKinds[0]?.kind || "");
      const historicRequests = metricHistory
        .map((snapshot) => snapshot.values.last_request)
        .filter((request): request is RuntimeRequestMetrics => Boolean(request?.id))
        .filter(
          (request, index, values) =>
            values.findIndex((candidate) => candidate.id === request.id) === index,
      );
      setRequestHistory(historicRequests.slice(-24).reverse());
      const currentContext = Number(status?.max_context);
      if (Number.isFinite(currentContext) && currentContext > 0) {
        setContextSize(Math.floor(currentContext));
      }
    } catch (cause) {
      if (!quiet) setError(errorMessage(cause));
    }
  }, []);

  useEffect(() => {
    if (!active || !runtime || voiceRef.current?.active) return;
    const key = `${runtime.model ?? model}:${active.mode}`;
    if (appliedModeTemplate.current === key) return;
    appliedModeTemplate.current = key;
    void voiceRef.current?.setFullDuplex(active.mode === "full_duplex");
  }, [active, model, runtime]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    voiceRef.current?.setPlayback(settings.playbackEnabled);
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === ",") {
        event.preventDefault();
        openStudioPage("dashboard", "settings");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    localStorage.setItem(STORED_PRESETS_KEY, JSON.stringify(storedPresets));
  }, [storedPresets]);

  useEffect(() => {
    const selected = jobKinds.find((item) => item.kind === selectedJobKind);
    const properties = selected?.payload_schema.properties ?? {};
    setJobPayload(
      Object.fromEntries(
        Object.entries(properties).map(([name, property]) => [
          name,
          selectedJobKind === "model.quantize" && name === "imatrix" && pendingImatrix
            ? pendingImatrix
            : schemaDefault(property),
        ]),
      ),
    );
  }, [jobKinds, selectedJobKind, pendingImatrix]);

  useEffect(() => {
    if (!selectedJobId) {
      setJobLogs([]);
      return;
    }
    void api.jobEvents(selectedJobId).then(setJobLogs).catch(() => undefined);
  }, [selectedJobId]);

  useEffect(() => {
    const stable = voiceMessages.filter(
      (message) => !message.pending && (message.text.trim() || message.audioId),
    );
    localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(stable.slice(-200)));
  }, [voiceMessages]);

  useEffect(() => {
    voiceRef.current = new RealtimeAudioController(
      {
        onState: setVoiceState,
        onLevel: setVoiceLevel,
        onText: (sessionId, text) =>
          setLiveVoice((current) =>
            text ? { sessionId, text } : current?.sessionId === sessionId ? null : current,
          ),
        onError: (message) => setError(message),
        onInputStart: ({ id, sessionId }) => {
          setVoiceMessages((current) => [
            ...current,
            {
              id,
              sessionId,
              role: "user",
              text: "",
              pending: true,
              created_at: new Date().toISOString(),
            },
          ]);
        },
        onInputEnd: ({ id, sessionId, audio }) => {
          const persist = async () => {
            if (!audio) {
              setVoiceMessages((current) => current.filter((message) => message.id !== id));
              return;
            }
            const audioId = `voice-${id}`;
            await saveVoiceClip(audioId, audio);
            setVoiceMessages((current) =>
              current.map((message) =>
                message.id === id && message.sessionId === sessionId
                  ? { ...message, audioId, pending: false }
                  : message,
              ),
            );
          };
          void persist().catch((cause) => setError(errorMessage(cause)));
        },
        onTurn: ({ id, sessionId, text, audio }) => {
          setVoiceMessages((current) => {
            const existing = current.find((message) => message.id === id);
            if (existing) {
              return current.map((message) =>
                message.id === id ? { ...message, text } : message,
              );
            }
            return [
              ...current,
              {
                id,
                sessionId,
                role: "assistant",
                text,
                created_at: new Date().toISOString(),
              },
            ];
          });
          if (!audio) return;
          const audioId = `voice-${id}`;
          const previous = voiceClipWrites.current.get(audioId) ?? Promise.resolve();
          const persist = previous
            .catch(() => undefined)
            .then(async () => {
              await saveVoiceClip(audioId, audio);
              setVoiceMessages((current) =>
                current.map((message) =>
                  message.id === id ? { ...message, audioId } : message,
                ),
              );
            });
          voiceClipWrites.current.set(audioId, persist);
          void persist
            .catch((cause) => setError(errorMessage(cause)))
            .finally(() => {
              if (voiceClipWrites.current.get(audioId) === persist) {
                voiceClipWrites.current.delete(audioId);
              }
            });
        },
      },
      settings.playbackEnabled,
      settings.fullDuplex,
    );
    return () => {
      void voiceRef.current?.stop();
      voiceRef.current = null;
    };
    // The controller reads current request settings when capture starts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let current = true;
    async function initialize() {
      try {
        let status = await studioStatus();
        let token: string | null = null;
        if (status) {
          try {
            token = await studioCredential();
          } catch (cause) {
            if (current) setError(errorMessage(cause));
          }
        }
        if (current && status) {
          setStudio(status);
          setStudioDraft({ ...status.config });
          if (token !== null) {
            setStudioToken(token);
            setStudioCredentialWritable(true);
          }
        }
        if (status?.config.mode === "local" && !status.reachable) {
          await startLocalStudio();
          status = await studioStatus();
          token = null;
          try {
            token = await studioCredential();
          } catch (cause) {
            if (current) setError(errorMessage(cause));
          }
        }
        if (status) {
          setApiBaseUrl(status.service_url);
          if (token !== null) setApiToken(token);
          if (current) {
            if (token !== null) {
              setStudioToken(token);
              setStudioCredentialWritable(true);
            }
            setStudio(status);
            setStudioDraft({ ...status.config });
          }
        }
        const initialStatusResult = await Promise.allSettled([api.runtimeStatus()]);
        const initialRuntime = initialStatusResult[0].status === "fulfilled"
          ? initialStatusResult[0].value
          : null;
        const runtimeReady = isRuntimeReady(initialRuntime?.runtime_state);
        const results = await Promise.allSettled([
          runtimeReady ? api.runtimeCapabilities() : Promise.resolve<RuntimeCapabilities | null>(null),
          runtimeReady ? api.runtimeModels() : Promise.resolve<RuntimeModel[]>([]),
          Promise.resolve(initialRuntime),
          runtimeReady ? api.realtimeCapabilities() : Promise.resolve<RealtimeCapabilities | null>(null),
          api.listSessions(),
          api.modelArtifacts(),
          api.runtimeInstances(),
          api.jobs(100),
          api.runtimeMetrics(200),
          api.runtimeLogs(100),
          api.jobKinds(),
          api.mcpServers(),
          api.mcpTools(),
          api.generationPresets(),
          api.runtimeProfiles(),
          api.artifactLineage(),
          api.datasets(),
          api.evaluations(),
          api.remoteNodes(),
          api.voiceOutputComponent(),
        ]);
        if (!current) return;
        if (results[0].status === "fulfilled") {
          setCapabilities(results[0].value);
        }
        if (results[1].status === "fulfilled") {
          const nextModels = results[1].value;
          setModels(nextModels);
        }
        if (results[2].status === "fulfilled") setRuntime(results[2].value);
        if (results[5].status === "fulfilled") setArtifacts(results[5].value);
        if (results[6].status === "fulfilled") setInstances(results[6].value);
        const initialInstances = results[6].status === "fulfilled" ? results[6].value : [];
        const initialModels = results[1].status === "fulfilled" ? results[1].value : [];
        const initialJobs = results[7].status === "fulfilled" ? results[7].value : [];
        const initialNames = runtimeSelectionNames(initialModels, initialInstances, initialJobs);
        const initialStatusModel = typeof initialRuntime?.model === "string"
          ? initialRuntime.model
          : "";
        setModel(
          initialStatusModel && initialNames.includes(initialStatusModel)
            ? initialStatusModel
            : initialNames[0] ?? "",
        );
        if (results[7].status === "fulfilled") setJobs(results[7].value);
        if (results[8].status === "fulfilled") {
          const history = results[8].value
            .map((snapshot) => snapshot.values.last_request)
            .filter((request): request is RuntimeRequestMetrics => Boolean(request?.id))
            .filter(
              (request, index, values) =>
                values.findIndex((candidate) => candidate.id === request.id) === index,
            );
          setRequestHistory(history.slice(-24).reverse());
        }
        if (results[9].status === "fulfilled") setRuntimeLogs(results[9].value);
        if (results[10].status === "fulfilled") {
          const kinds = results[10].value;
          setJobKinds(kinds);
          setSelectedJobKind((current) => current || kinds[0]?.kind || "");
        }
        if (results[11].status === "fulfilled") setMcpServers(results[11].value);
        if (results[12].status === "fulfilled") setMcpTools(results[12].value.data);
        if (results[13].status === "fulfilled") {
          if (results[13].value.length > 0) {
            setStoredPresets(results[13].value.map(storedPresetFromResource));
          }
        }
        if (results[14].status === "fulfilled") setRuntimeProfiles(results[14].value);
        if (results[15].status === "fulfilled") setLineage(results[15].value);
        if (results[16].status === "fulfilled") setDatasets(results[16].value);
        if (results[17].status === "fulfilled") setEvaluations(results[17].value);
        if (results[18].status === "fulfilled") setRemoteNodes(results[18].value);
        if (results[19].status === "fulfilled") setVoiceComponent(results[19].value);
        if (results[3].status === "fulfilled") {
          setRealtime(results[3].value);
          setRealtimeAvailable(results[3].value?.available === true);
        }
        if (results[4].status === "fulfilled") {
          setSessions(results[4].value);
          setActiveId(results[4].value[0]?.id ?? null);
        } else {
          throw results[4].reason;
        }
      } catch (cause) {
        if (current) setError(errorMessage(cause));
      }
    }
    void initialize();
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    if (!activeJobIds) return;
    const controller = new AbortController();
    for (const id of activeJobIds.split(",")) {
      void api.streamJobEvents(id, (event) => {
        const status = event.type === "state" && typeof event.data.status === "string"
          ? event.data.status as JobResource["status"]
          : null;
        setJobs((current) => current.map((job) => job.id === id ? {
          ...job,
          ...(status ? { status } : {}),
          ...(typeof event.progress === "number" ? { progress: event.progress } : {}),
          updated_at: event.created_at,
        } : job));
        if (event.job_id === selectedJobId && event.message) {
          setJobLogs((current) => current.some((entry) => entry.sequence === event.sequence)
            ? current
            : [...current, {
              sequence: event.sequence,
              level: event.level,
              message: event.message || "",
              fields: event.data,
              created_at: event.created_at,
            }]);
        }
        if (status && ["succeeded", "failed", "cancelled", "interrupted"].includes(status)) {
          void refreshRuntime(true);
        }
      }, controller.signal).catch((cause) => {
        if (!controller.signal.aborted) setError(errorMessage(cause));
      });
    }
    return () => controller.abort();
  }, [activeJobIds, refreshRuntime, selectedJobId]);

  useEffect(() => {
    const instanceId = selectedRuntimeInstance?.id;
    if (!instanceId || !isRuntimeReady(selectedRuntimeInstance?.state)) return;
    let current = true;
    void Promise.allSettled([
      api.runtimeCapabilities(instanceId),
      api.runtimeStatus(instanceId),
    ]).then(([capabilityResult, statusResult]) => {
      if (!current) return;
      if (capabilityResult.status === "fulfilled") {
        setCapabilities(capabilityResult.value);
      }
      if (statusResult.status === "fulfilled") {
        setRuntime(statusResult.value);
        const nextContext = Number(statusResult.value.max_context);
        if (Number.isFinite(nextContext) && nextContext > 0) {
          setContextSize(Math.floor(nextContext));
        }
      }
    });
    return () => {
      current = false;
    };
  }, [selectedRuntimeInstance?.id]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setResponses({});
      return;
    }
    let current = true;
    Promise.all([api.listMessages(activeId), api.listResponses(activeId)])
      .then(([nextMessages, nextResponses]) => {
        if (!current) return;
        setMessages(nextMessages);
        setResponses(
          Object.fromEntries(
            nextResponses
              .filter((response) => response.output_message_id)
              .map((response) => [response.output_message_id as string, response]),
          ),
        );
      })
      .catch((cause) => current && setError(errorMessage(cause)));
    return () => {
      current = false;
      if (voiceRef.current?.active) void voiceRef.current.stop();
    };
  }, [activeId]);

  const activeSessionId = active?.id;
  const activeSessionModel = active?.model;
  const activeSessionTitle = active?.title;
  useEffect(() => {
    if (!activeSessionId || !selectedModelAvailable || activeSessionModel === model) return;
    const switchKey = `${activeSessionId}:${model}`;
    if (sessionSwitchRef.current === switchKey) return;
    sessionSwitchRef.current = switchKey;
    let current = true;
    setSessionTransitioning(true);
    void api.forkSession(activeSessionId, null, true, activeSessionTitle, model)
      .then((replacement) => {
        if (!current) {
          void refreshSessions();
          return;
        }
        setSessions((existing) => [replacement, ...existing]);
        setActiveId(replacement.id);
        setResponses({});
      })
      .catch((cause) => {
        if (current) setError(errorMessage(cause));
      })
      .finally(() => {
        setSessionTransitioning(false);
        if (sessionSwitchRef.current === switchKey) sessionSwitchRef.current = "";
      });
    return () => {
      current = false;
    };
  }, [activeSessionId, activeSessionModel, activeSessionTitle, model, refreshSessions, selectedModelAvailable]);

  useEffect(() => {
    setAttachments((current) => {
      current.forEach((attachment) => {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      });
      return [];
    });
  }, [activeId]);

  const handleMessageScroll = useCallback(() => {
    const scroller = messageScrollerRef.current;
    if (!scroller) return;
    const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    autoFollowOutputRef.current = distanceFromBottom <= 8;
  }, []);

  useEffect(() => {
    autoFollowOutputRef.current = true;
  }, [activeId]);

  useEffect(() => {
    const scroller = messageScrollerRef.current;
    if (!scroller || !autoFollowOutputRef.current) return;
    const frame = window.requestAnimationFrame(() => {
      scroller.scrollTop = scroller.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, currentVoiceMessages, liveVoice, live, busy]);

  useEffect(() => {
    if (capabilities && !capabilities.model_capabilities.features.full_duplex) {
      setMode("text");
    }
  }, [capabilities]);

  function samplingParams(): SamplingParams {
    return {
      max_tokens: effectiveSettings.maxTokens,
      temperature: effectiveSettings.temperature,
      top_k: effectiveSettings.topK,
      top_p: effectiveSettings.topP,
      presence_penalty: effectiveSettings.presencePenalty,
      frequency_penalty: effectiveSettings.frequencyPenalty,
      repetition_penalty: effectiveSettings.repetitionPenalty,
      seed: effectiveSettings.seed,
      enable_thinking: thinkingSupported && effectiveSettings.enableThinking,
      enable_vision: effectiveSettings.enableVision,
      enable_mtp: mtpSupported && mtpAvailable && effectiveSettings.enableMtp,
      reasoning_effort: effectiveSettings.reasoningEffort || null,
    };
  }

  function realtimeSessionConfig(sessionId: string) {
    return {
      sessionId,
      systemPrompt: effectiveSystemPrompt,
      temperature: effectiveSettings.temperature,
      topP: effectiveSettings.topP,
      topK: effectiveSettings.topK,
      repetitionPenalty: effectiveSettings.repetitionPenalty,
    };
  }

  function openStudioPage(nextView: "dashboard" | "lab", page: DashboardPage | LabPage) {
    navigate(nextView === 'dashboard'
      ? dashboardPath(page as DashboardPage)
      : labPath(page as LabPage));
    closeSidebar();
  }

  /** 打开对话页并关闭移动端侧栏。 */
  function openChatPage() {
    navigate(STUDIO_PATHS.chat);
    closeSidebar();
  }

  function selectModel(nextModel: string) {
    if (!nextModel || nextModel === model || sessionTransitioning) return;
    if (active) setSessionTransitioning(true);
    setModel(nextModel);
  }

  function selectSession(id: string) {
    const session = sessions.find((candidate) => candidate.id === id);
    if (!session || session.id === activeId || busy || sessionTransitioning) return;
    if (voiceRef.current?.active) void voiceRef.current.stop();
    setModel(session.model);
    setMode(session.mode);
    setActiveId(session.id);
    setMessages([]);
    setResponses({});
    openChatPage();
  }

  async function createSession() {
    const selectedModel = model.trim();
    if (!selectedModel || sessionTransitioning) return;
    setSessionTransitioning(true);
    setError(null);
    try {
      const created = await api.createSession(selectedModel, mode);
      setSessions((current) => [created, ...current]);
      setActiveId(created.id);
      setMessages([]);
      openChatPage();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSessionTransitioning(false);
    }
  }

  async function clearActiveConversation() {
    if (!active || !conversationReady || busy || !await studioConfirm(tr("清空当前对话？", "Clear this conversation?"))) return;
    setBusy(true);
    setError(null);
    try {
      const replacement = await api.createSession(active.model, active.mode);
      await api.deleteSession(active.id);
      setSessions((current) => [replacement, ...current.filter((session) => session.id !== active.id)]);
      setVoiceMessages((current) => current.filter((message) => message.sessionId !== active.id));
      setMessages([]);
      setResponses({});
      setActiveId(replacement.id);
    } catch (cause) {
      setError(errorMessage(cause));
      await refreshSessions();
    } finally {
      setBusy(false);
    }
  }

  function applyFrame(frame: RealtimeFrame) {
    const payload = frame.payload;
    if (payload.type === "session.state") {
      setSessions((current) =>
        current.map((session) =>
          session.id === frame.session_id
            ? {
                ...session,
                state: payload.state as Session["state"],
                revision: payload.revision as number,
              }
            : session,
        ),
      );
      return;
    }
    if (payload.type === "response.reasoning.delta") {
      setLive((current) => ({
        reasoning: (current?.reasoning ?? "") + String(payload.delta ?? ""),
        text: current?.text ?? "",
        tools: current?.tools ?? [],
      }));
    } else if (payload.type === "response.text.delta") {
      setLive((current) => ({
        reasoning: current?.reasoning ?? "",
        text: (current?.text ?? "") + String(payload.delta ?? ""),
        tools: current?.tools ?? [],
      }));
    } else if (payload.type === "response.tool_call.delta") {
      setLive((current) => {
        const tools = [...(current?.tools ?? [])];
        const index = Number(payload.index);
        tools[index] = (tools[index] ?? "") + String(payload.arguments_delta ?? "");
        return {
          reasoning: current?.reasoning ?? "",
          text: current?.text ?? "",
          tools,
        };
      });
    }
  }

  async function generate(
    session: Session,
    input: ContentPart[],
    optimistic = true,
    inputRole: "user" | "tool" = "user",
  ) {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setBusy(true);
    setStopping(false);
    setLive({ reasoning: "", text: "", tools: [] });
    if (optimistic) {
      if (inputRole !== "user") throw new Error("Only user input can be optimistic");
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "user",
          parts: input,
          parent_id: current.at(-1)?.id ?? null,
          created_at: new Date().toISOString(),
        },
      ]);
    }
    try {
      await streamResponse(
        session.id,
        {
          request_id: crypto.randomUUID(),
          expected_revision: session.revision,
          input,
          input_role: inputRole,
          sampling: samplingParams(),
          system_prompt: effectiveSystemPrompt,
          include_reasoning_history: !effectiveSettings.excludeReasoning,
          tools: mcpTools
            .filter((tool) => selectedTools.includes(tool.qualified_name))
            .map((tool) => ({
              type: "function" as const,
              function: {
                name: tool.qualified_name,
                description: tool.description,
                parameters: tool.input_schema,
              },
            })),
          tool_choice: selectedTools.length ? "auto" : "none",
          stream: true,
        },
        applyFrame,
        controller.signal,
      );
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(errorMessage(cause));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
      setStopping(false);
      setLive(null);
      try {
        const [persisted, persistedResponses, updated] = await Promise.all([
          api.listMessages(session.id),
          api.listResponses(session.id),
          api.getSession(session.id),
        ]);
        setMessages(persisted);
        setResponses(
          Object.fromEntries(
            persistedResponses
              .filter((response) => response.output_message_id)
              .map((response) => [response.output_message_id as string, response]),
          ),
        );
        await refreshSessions(updated.id);
      } catch (cause) {
        setError(errorMessage(cause));
      }
      void refreshRuntime(true);
    }
  }

  async function stopGeneration() {
    const controller = abortRef.current;
    if (!active || !controller || stopping) return;
    setStopping(true);
    try {
      await api.cancelResponse(active.id);
    } catch (cause) {
      if (!(cause instanceof ApiError && cause.status === 409)) {
        setError(errorMessage(cause));
      }
    } finally {
      controller.abort();
      setStopping(false);
    }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!active || !conversationReady || (!text && !attachments.length) || busy) return;
    if (
      active.mode !== "text" &&
      realtimeAvailable &&
      voiceRef.current &&
      attachments.length === 0
    ) {
      if (!text) return;
      setDraft("");
      await voiceRef.current.submitText(text, realtimeSessionConfig(active.id));
      setVoiceMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          sessionId: active.id,
          role: "user",
          text,
          created_at: new Date().toISOString(),
        },
      ]);
      return;
    }
    const realtimeController = voiceRef.current;
    const resumeCapture = Boolean(realtimeController?.capturing);
    if (active.mode !== "text" && realtimeController?.active) {
      await realtimeController.stop();
    }
    setBusy(true);
    setError(null);
    try {
      const uploaded = await Promise.all(
        attachments.map(async (attachment): Promise<ContentPart> => {
          if (attachment.kind === "document") {
            const uploaded = await api.uploadMedia(
              attachment.file,
              documentMimeType(attachment.file),
            );
            const document = await api.createDocument(uploaded.media.id, attachment.file.name);
            return { type: "document", media: document.media, name: document.name };
          }
          const metadata = await mediaMetadata(attachment.file, attachment.kind);
          const resource = await api.uploadMedia(attachment.file);
          if (attachment.kind === "image") {
            const image = metadata as { width: number; height: number };
            return { type: "image", media: resource.media, ...image };
          }
          if (attachment.kind === "video") {
            const video = metadata as { width: number; height: number; duration_ms: number };
            return { type: "video", media: resource.media, ...video };
          }
          const audio = metadata as {
            sample_rate_hz: number;
            channels: number;
            duration_ms: number;
          };
          return { type: "audio", media: resource.media, ...audio };
        }),
      );
      const input: ContentPart[] = [...uploaded];
      if (text) input.push({ type: "text", text });
      setDraft("");
      setAttachments((current) => {
        current.forEach((attachment) => {
          if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
        });
        return [];
      });
      await generate(active, input);
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    } finally {
      if (resumeCapture && realtimeController && realtimeAvailable) {
        try {
          await realtimeController.start(realtimeSessionConfig(active.id));
        } catch (cause) {
          setError(errorMessage(cause));
        }
      }
    }
  }

  function selectAttachments(files: FileList | null) {
    if (!files) return;
    const next: PendingAttachment[] = [];
    for (const file of Array.from(files).slice(0, 8)) {
      const kind = file.type.startsWith("image/")
        ? "image"
        : file.type.startsWith("video/")
          ? "video"
          : file.type.startsWith("audio/")
            ? "audio"
            : isTextDocument(file)
              ? "document"
              : null;
      if (!kind) continue;
      if (kind === "document") {
        if (file.size > MAX_DOCUMENT_BYTES) {
          setError(tr("文档不能超过 64 MiB。", "Documents are limited to 64 MiB."));
          continue;
        }
      } else if (!attachmentAccept.includes(`${kind}/*`)) {
        continue;
      }
      next.push({
        id: crypto.randomUUID(),
        file,
        kind,
        previewUrl: kind === "document" ? "" : URL.createObjectURL(file),
      });
    }
    setAttachments((current) => {
      const combined = [...current, ...next];
      combined.slice(8).forEach((attachment) => {
        if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
      });
      return combined.slice(0, 8);
    });
    if (attachmentInputRef.current) attachmentInputRef.current.value = "";
  }

  function removeAttachment(id: string) {
    setAttachments((current) => {
      const removed = current.find((attachment) => attachment.id === id);
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      return current.filter((attachment) => attachment.id !== id);
    });
  }

  async function saveEdit(message: Message) {
    if (!active || !conversationReady || !editDraft || busy) return;
    const messageIndex = messages.findIndex((item) => item.id === message.id);
    if (messageIndex < 0) return;
    const text = editDraft.text.trim();
    if (!text && !message.parts.some(isMediaPart)) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(
        active.id,
        active.revision,
        message.id,
        false,
      );
      const parts: ContentPart[] = [];
      if (text) parts.push({ type: "text", text });
      parts.push(...message.parts.filter(isMediaPart));
      const rewoundMessages = messages.slice(0, messageIndex);
      const visibleMessageIds = new Set(rewoundMessages.map((item) => item.id));
      setSessions((current) =>
        current.map((session) => session.id === rewound.id ? rewound : session),
      );
      setMessages(rewoundMessages);
      setResponses((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([messageId]) => visibleMessageIds.has(messageId)),
        ),
      );
      setEditDraft(null);
      setMessages([...rewoundMessages, { ...message, parts }]);
      await generate(rewound, parts, false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function regenerate(message: Message) {
    if (!active || !conversationReady || busy || message.role !== "assistant") return;
    const index = messages.findIndex((item) => item.id === message.id);
    let user: Message | undefined;
    let userIndex = -1;
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (messages[cursor].role === "user") {
        user = messages[cursor];
        userIndex = cursor;
        break;
      }
    }
    if (!user) return;
    if (!user.parts.length) return;
    setBusy(true);
    try {
      const rewound = await api.rewindSession(
        active.id,
        active.revision,
        user.id,
        false,
      );
      const rewoundMessages = messages.slice(0, userIndex + 1);
      const visibleMessageIds = new Set(rewoundMessages.map((item) => item.id));
      setSessions((current) =>
        current.map((session) => session.id === rewound.id ? rewound : session),
      );
      setMessages(rewoundMessages);
      setResponses((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([messageId]) => visibleMessageIds.has(messageId)),
        ),
      );
      await generate(rewound, user.parts, false);
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    }
  }

  async function copyMessage(message: Message) {
    const parts = textParts(message);
    await navigator.clipboard.writeText([parts.reasoning, parts.text].filter(Boolean).join("\n\n"));
  }

  async function copyEndpoint() {
    try {
      await navigator.clipboard.writeText(studio?.service_url || "http://127.0.0.1:8090");
      setEndpointCopied(true);
      window.setTimeout(() => setEndpointCopied(false), 1600);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function executeToolCalls(message: Message) {
    if (!active || !conversationReady || busy) return;
    const calls = message.parts.filter(
      (part): part is Extract<ContentPart, { type: "tool_call" }> => part.type === "tool_call",
    );
    if (!calls.length) return;
    setBusy(true);
    setError(null);
    try {
      const results = await Promise.all(
        calls.map(async (call) => ({
          call,
          result: await api.callMcpTool(call.name, call.arguments),
        })),
      );
      let updated = await api.getSession(active.id);
      for (const item of results.slice(0, -1)) {
        const appended = await api.appendMessage(updated.id, updated.revision, "tool", [{
          type: "tool_result",
          call_id: item.call.call_id,
          result: item.result.structured_content ?? item.result.content,
          is_error: item.result.is_error,
        }]);
        updated = appended.session;
      }
      const last = results.at(-1)!;
      setBusy(false);
      await generate(
        updated,
        [{
          type: "tool_result",
          call_id: last.call.call_id,
          result: last.result.structured_content ?? last.result.content,
          is_error: last.result.is_error,
        }],
        false,
        "tool",
      );
    } catch (cause) {
      setBusy(false);
      setError(errorMessage(cause));
    }
  }

  async function createMcpServer(event: FormEvent) {
    event.preventDefault();
    if (!mcpDraft.name.trim() || !mcpDraft.endpoint.trim() || busy) return;
    setBusy(true);
    try {
      await api.createMcpServer({
        name: mcpDraft.name.trim(),
        transport: mcpDraft.transport,
        enabled: true,
        url: mcpDraft.transport === "streamable_http" ? mcpDraft.endpoint.trim() : null,
        command: mcpDraft.transport === "stdio" ? mcpDraft.endpoint.trim() : null,
      });
      setMcpDraft({ name: "", transport: "streamable_http", endpoint: "" });
      await Promise.all([refreshRuntime(false), refreshMcp()]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function toggleMcpServer(server: McpServerResource) {
    try {
      await api.updateMcpServer(server.id, !server.enabled);
      await refreshMcp();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function deleteMcpServer(id: string) {
    try {
      await api.deleteMcpServer(id);
      await refreshMcp();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function toggleVoice() {
    if (
      !active ||
      active.mode === "text" ||
      !realtimeAvailable ||
      busy ||
      !voiceRef.current
    ) return;
    await voiceRef.current.toggleCapture(realtimeSessionConfig(active.id));
  }

  async function installOrEnableVoiceOutput() {
    if (voiceComponentBusy) return;
    setVoiceComponentBusy(true);
    try {
      if (voiceComponent?.ready) {
        const result = await api.activateVoiceOutputComponent();
        if (!result.active) {
          throw new Error(result.error || result.reason || tr("语音输出启用失败", "Voice output activation failed"));
        }
      } else {
        const accepted = await api.installVoiceOutputComponent();
        setSelectedJobId(accepted.operation_id);
        setVoiceComponent((current) => current && ({ ...current, state: "installing" }));
      }
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setVoiceComponentBusy(false);
    }
  }

  async function selectInteractionMode(nextMode: SessionMode) {
    setMode(nextMode);
    if (!active || active.mode === nextMode || busy) return;
    if (voiceRef.current?.active) await voiceRef.current.stop();
    try {
      const updated = await api.updateSession(active.id, { mode: nextMode });
      appliedModeTemplate.current = "";
      setSessions((current) =>
        current.map((session) => (session.id === updated.id ? updated : session)),
      );
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  function openSettings() {
    setSettingsDraft(resolvedGlobalSettings);
    const current = Number(runtime?.max_context);
    setContextSize(Number.isFinite(current) && current > 0 ? current : contextSize);
    setPresetStatus(null);
    openStudioPage("dashboard", "settings");
  }

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeSidebar();
      } else if ((event.metaKey || event.ctrlKey) && event.key === ",") {
        event.preventDefault();
        openSettings();
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  });

  function applyPreset(name: Exclude<PresetName, "custom">) {
    setSettingsDraft((current) => ({
      ...current,
      ...PRESETS[name],
      inheritModelDefaults: false,
      preset: name,
    }));
    setSelectedStoredPreset("");
    setStoredPresetName("");
    setPresetStatus(null);
  }

  function loadStoredPreset(name: string) {
    setSelectedStoredPreset(name);
    setPresetStatus(null);
    if (!name) {
      setStoredPresetName("");
      return;
    }
    const stored = storedPresets.find((preset) => preset.name === name);
    if (!stored) return;
    setStoredPresetName(stored.name);
    setSettingsDraft((current) => ({
      ...current,
      ...stored.settings,
      inheritModelDefaults: false,
      preset: "custom",
    }));
    setContextSize(stored.contextSize);
    setPresetStatus({ error: false, text: tr("预设已载入。", "Preset loaded.") });
  }

  async function saveStoredPreset() {
    const name = storedPresetName.replace(/\s+/g, " ").trim().slice(0, 64);
    if (!name) {
      setPresetStatus({ error: true, text: tr("请输入预设名称。", "Enter a preset name.") });
      return;
    }
    const next: StoredPreset = {
      name,
      settings: presetSnapshot(settingsDraft),
      inheritGlobalSettings: false,
      contextSize,
      model: active?.model ?? model,
      mode: active?.mode ?? mode,
      updatedAt: new Date().toISOString(),
    };
    const selected = storedPresets.find((preset) => preset.name === selectedStoredPreset);
    const body = {
      name,
      model: next.model,
      mode: next.mode,
      settings: {
        sampling: {
          max_tokens: next.settings.maxTokens,
          temperature: next.settings.temperature,
          top_k: next.settings.topK,
          top_p: next.settings.topP,
          presence_penalty: next.settings.presencePenalty,
          frequency_penalty: next.settings.frequencyPenalty,
          repetition_penalty: next.settings.repetitionPenalty,
          seed: next.settings.seed,
          enable_thinking: thinkingSupported && next.settings.enableThinking,
          enable_vision: next.settings.enableVision,
          enable_mtp: next.settings.enableMtp,
          reasoning_effort: next.settings.reasoningEffort || null,
        },
        system_prompt: next.settings.systemPrompt || null,
        include_reasoning_history: !next.settings.excludeReasoning,
        input_role: "user" as const,
        tools: [],
        tool_choice: "auto" as const,
        response_format: { type: "text" as const },
      },
      context_size: next.contextSize,
      metadata: {
        ...(selected?.icon ? { icon: selected.icon } : {}),
        inherit_global_settings: false,
      },
    };
    try {
      const saved = selected?.id
        ? await api.updateGenerationPreset(selected.id, body)
        : await api.createGenerationPreset(body);
      const stored = storedPresetFromResource(saved);
      setStoredPresets((current) => {
        const index = current.findIndex(
          (preset) => preset.id === stored.id || preset.name === selectedStoredPreset,
        );
        if (index < 0) return [...current, stored].slice(-50);
        return current.map((preset, cursor) => cursor === index ? stored : preset);
      });
      setSelectedStoredPreset(stored.name);
      setStoredPresetName(stored.name);
      setPresetStatus({ error: false, text: tr("预设已保存。", "Preset saved.") });
    } catch (cause) {
      setPresetStatus({ error: true, text: errorMessage(cause) });
    }
  }

  async function deleteStoredPreset() {
    if (!selectedStoredPreset) return;
    if (!await studioConfirm(tr(`删除预设“${selectedStoredPreset}”？`, `Delete preset “${selectedStoredPreset}”?`))) return;
    const selected = storedPresets.find((preset) => preset.name === selectedStoredPreset);
    try {
      if (selected?.id) await api.deleteGenerationPreset(selected.id);
      setStoredPresets((current) =>
        current.filter((preset) => preset.name !== selectedStoredPreset),
      );
      setSelectedStoredPreset("");
      setStoredPresetName("");
      setPresetStatus({ error: false, text: tr("预设已删除。", "Preset deleted.") });
    } catch (cause) {
      setPresetStatus({ error: true, text: errorMessage(cause) });
    }
  }

  function renderPresetManager(disabled = false) {
    return (
      <div className="saved-presets">
        <label>
          <span>{tr("已保存预设", "Saved presets")}</span>
          <select
            disabled={disabled}
            onChange={(event) => loadStoredPreset(event.target.value)}
            value={selectedStoredPreset}
          >
            <option value="">
              {storedPresets.length
                ? tr("选择预设…", "Select a preset…")
                : tr("还没有保存的预设", "No saved presets")}
            </option>
            {storedPresets.map((preset) => (
              <option key={preset.name} value={preset.name}>{preset.name}</option>
            ))}
          </select>
        </label>
        <div className="preset-save-row">
          <input
            aria-label={tr("预设名称", "Preset name")}
            disabled={disabled}
            maxLength={64}
            onChange={(event) => {
              setStoredPresetName(event.target.value);
              setPresetStatus(null);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void saveStoredPreset();
              }
            }}
            placeholder={tr("预设名称", "Preset name")}
            value={storedPresetName}
          />
          <button disabled={disabled} onClick={() => void saveStoredPreset()} type="button">
            {selectedStoredPreset && storedPresetName.trim() === selectedStoredPreset
              ? tr("覆盖", "Update")
              : tr("保存", "Save")}
          </button>
          <button
            aria-label={tr("删除预设", "Delete preset")}
            className="preset-delete"
            disabled={disabled || !selectedStoredPreset}
            onClick={() => void deleteStoredPreset()}
            title={tr("删除预设", "Delete preset")}
            type="button"
          >
            <Icon name="trash" size={14} />
          </button>
        </div>
        {presetStatus && (
          <p className={presetStatus.error ? "preset-status error" : "preset-status"}>
            {presetStatus.text}
          </p>
        )}
        <small>
          {tr(
            "保存系统提示词、上下文和生成参数；不包含界面语言与播放开关。",
            "Stores the system prompt, context, and generation parameters; interface and playback preferences stay separate.",
          )}
        </small>
      </div>
    );
  }

  const presetManager = renderPresetManager(settingsDraft.inheritModelDefaults);

  function setModelDefaultInheritance(enabled: boolean) {
    setSettingsDraft((current) => ({
      ...(enabled
        ? modeTemplateSettings(current, active?.mode ?? mode, runtime, realtime)
        : current),
      inheritModelDefaults: enabled,
    }));
    if (enabled) {
      setSelectedStoredPreset("");
      setStoredPresetName("");
      setPresetStatus(null);
    }
  }

  function saveSettings() {
    setSettings(settingsDraft);
  }

  function updateGlobalInference(patch: Partial<GenerationSettings>) {
    setSettings((current) => ({
      ...(current.inheritModelDefaults
        ? modeTemplateSettings(current, active?.mode ?? mode, runtime, realtime)
        : current),
      ...patch,
      inheritModelDefaults: false,
      preset: "custom",
    }));
  }

  async function reloadRuntime() {
    if (busy || voiceRef.current?.active) return;
    if (!await studioConfirm(tr(`以 ${formatNumber(contextSize)} token 上下文重载模型？`, `Reload the model with a ${formatNumber(contextSize)} token context?`))) return;
    setBusy(true);
    try {
      const status = await api.reloadRuntime(contextSize, runtime?.instance_id);
      setRuntime(status);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function clearRuntimeCache() {
    const snapshots = Number(runtime?.prefix_cache_snapshots || 0);
    if (busy || snapshots <= 0 || Number(runtime?.active_requests || 0) > 0) return;
    const hotPrefixOnly = runtime?.prefix_cache_mode === "single_device_hot_prefix";
    const confirmation = hotPrefixOnly
      ? tr(
          `清除当前设备热前缀（${formatNumber(snapshots)} 个）？此操作不会删除聊天记录。`,
          `Clear ${formatNumber(snapshots)} device-hot prefix? Chat history will be kept.`,
        )
      : tr(
          `清除 ${formatNumber(snapshots)} 个 Session KV SSD 缓存块？此操作不会删除聊天记录。`,
          `Clear ${formatNumber(snapshots)} Session KV SSD cache blocks? Chat history will be kept.`,
        );
    if (!await studioConfirm(confirmation)) return;
    setBusy(true);
    try {
      const status = await api.clearRuntimeCache(runtime?.instance_id);
      setRuntime((current) => current ? { ...current, ...status } : status);
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function openServerPage() {
    if (studio) setStudioDraft({ ...studio.config });
    openStudioPage("dashboard", "connections");
  }

  async function saveStudioSettings() {
    if (!studioDraft || sessionTransitioning) return;
    setBusy(true);
    try {
      const status = await configureStudio(studioDraft);
      if (studioCredentialWritable) await saveStudioCredential(studioToken);
      setApiBaseUrl(status.service_url);
      setApiToken(studioToken);
      setStudio(status);
      setStudioDraft({ ...status.config });
      setMessages([]);
      setActiveId(null);
      await refreshSessions();
      appliedModeTemplate.current = "";
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function updateJobPayload(name: string, property: JsonSchemaProperty, value: string | boolean) {
    const type = schemaType(property);
    let parsed: unknown = value;
    if (type === "integer" || type === "number") {
      parsed = value === "" ? null : Number(value);
    } else if (type === "array") {
      parsed = String(value)
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => (property.items?.type === "integer" ? Number(item) : item));
    }
    setJobPayload((current) => ({ ...current, [name]: parsed }));
  }

  async function submitJob(event: FormEvent) {
    event.preventDefault();
    if (!selectedJobKind) return;
    setBusy(true);
    try {
      const clean = Object.fromEntries(
        Object.entries(jobPayload).filter(([, value]) => value !== "" && value !== null),
      );
      const created = await api.createJob(selectedJobKind, clean);
      setSelectedJobId(created.id);
      setJobs((current) => [created, ...current]);
      navigate(STUDIO_PATHS.quantization);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function importImatrix(files: FileList | null) {
    const file = files?.[0];
    if (!file || imatrixImporting || busy) return;
    setImatrixImporting(true);
    try {
      const uploaded = await api.uploadMedia(file, "application/x-mfq-imatrix");
      const destination = `artifacts/imatrix/${file.name.replace(/[^A-Za-z0-9_.-]+/g, "-")}`;
      const created = await api.createJob("artifact.import", {
        media_id: uploaded.media.id,
        destination,
        kind: "imatrix",
      });
      setSelectedJobId(created.id);
      setJobs((current) => [created, ...current]);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImatrixImporting(false);
      if (imatrixInputRef.current) imatrixInputRef.current.value = "";
    }
  }


  async function cancelSelectedJob() {
    if (!selectedJobId) return;
    try {
      const updated = await api.cancelJob(selectedJobId);
      setJobs((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function retrySelectedJob() {
    if (!selectedJob || busy) return;
    setBusy(true);
    try {
      const retried = await api.retryJob(selectedJob.id);
      setJobs((current) => [retried, ...current]);
      setSelectedJobId(retried.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function deleteJobRecord(id: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.deleteJob(id);
      setJobs((current) => current.filter((item) => item.id !== id));
      if (selectedJobId === id) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  async function clearCompletedJobRecords() {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.clearCompletedJobs();
      setJobs((current) => current.filter((item) => !isTerminalJob(item)));
      if (selectedJob && isTerminalJob(selectedJob)) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  async function searchHub(event: FormEvent) {
    event.preventDefault();
    const query = hubQuery.trim();
    if (!query || busy) return;
    setBusy(true);
    try {
      const reference = parseHubReference(query, hubProvider);
      if (reference) {
        const info = await api.hubModelInfo(
          reference.provider,
          reference.repoId,
          reference.revision,
        );
        setHubProvider(reference.provider);
        setHubResults([info]);
        setHubModel(info);
        return;
      }
      const results = await api.searchHubModels(hubProvider, query);
      setHubResults(results);
      setHubModel(null);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function inspectHubModel(item: HubModelSummary) {
    if (busy) return;
    setBusy(true);
    try {
      setHubModel(await api.hubModelInfo(item.provider, item.repo_id));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function downloadHubModel() {
    if (!hubModel || busy) return;
    const name = hubModel.repo_id.split("/").pop()?.replace(/[^A-Za-z0-9_.-]/g, "-") || "model";
    const repositoryPath = hubModel.repo_id
      .split("/")
      .map((part) => part.replace(/[^A-Za-z0-9_.-]+/g, "-") || "model")
      .join("/");
    setBusy(true);
    try {
      const created = await api.createJob(
        `download.${hubModel.provider}`,
        {
          repo_id: hubModel.repo_id,
          destination: `models/${hubModel.provider}/${repositoryPath || name}`,
          revision: hubModel.revision,
          expected_bytes: hubModel.total_bytes || null,
        },
      );
      setJobs((current) => [created, ...current]);
      setSelectedJobId(created.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function removeSelectedArtifact() {
    const uri = String(selectedJob?.result?.artifact || "");
    if (!uri.startsWith("workspace://") || busy) return;
    if (!await studioConfirm(tr("删除这个本地产物？", "Delete this local artifact?"))) return;
    setBusy(true);
    try {
      await api.removeWorkspaceArtifact(uri);
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerDataset(event: FormEvent) {
    event.preventDefault();
    if (busy || !datasetDraft.name.trim() || !datasetDraft.artifact_uri.trim()) return;
    setBusy(true);
    try {
      await api.createDataset({
        name: datasetDraft.name.trim(),
        kind: datasetDraft.kind,
        artifact_uri: datasetDraft.artifact_uri.trim(),
      });
      setDatasetDraft({ name: "", artifact_uri: "", kind: "custom" });
      setDatasets(await api.datasets());
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerRemoteNode(event: FormEvent) {
    event.preventDefault();
    if (busy || !nodeDraft.name.trim() || !nodeDraft.url.trim()) return;
    setBusy(true);
    try {
      await api.createRemoteNode({
        name: nodeDraft.name.trim(),
        url: nodeDraft.url.trim(),
        api_key_env: nodeDraft.api_key_env.trim() || null,
        enabled: true,
      });
      setNodeDraft({ name: "", url: "", api_key_env: "" });
      setRemoteNodes(await api.remoteNodes(true));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function compareSelectedEvaluations() {
    if (selectedEvaluations.length < 2 || busy) return;
    setBusy(true);
    try {
      setEvaluationComparison(await api.compareEvaluations(selectedEvaluations));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function exportStudioData() {
    try {
      const sessionArchives = await Promise.all(sessions.map((session) => api.exportSession(session.id)));
      const payload = {
        format: "mfq-studio-export-v2",
        exported_at: new Date().toISOString(),
        sessions: sessionArchives,
        presets: storedPresets,
      };
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
      anchor.download = `mfq-studio-${new Date().toISOString().slice(0, 10)}.json`;
      anchor.click();
      URL.revokeObjectURL(anchor.href);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function importStudioData(file: File) {
    try {
      const payload = JSON.parse(await file.text()) as { format?: string; presets?: StoredPreset[]; sessions?: SessionArchive[] };
      if (payload.format !== "mfq-studio-export-v2" || !Array.isArray(payload.presets) || !Array.isArray(payload.sessions)) {
        throw new Error(tr("不是有效的 MFQ Studio 导出文件。", "Not a valid MFQ Studio export."));
      }
      for (const preset of payload.presets) {
        if (!preset.name || !preset.settings || !Number.isFinite(preset.contextSize)) continue;
        const existing = storedPresets.find((item) => item.name === preset.name);
        const resource = presetResourceBody({
          ...preset,
          id: existing?.id,
          inheritGlobalSettings: preset.inheritGlobalSettings !== false,
        }, model, mode);
        if (existing?.id) await api.updateGenerationPreset(existing.id, resource);
        else await api.createGenerationPreset(resource);
      }
      for (const archive of payload.sessions) await api.importSession(archive);
      const next = (await api.generationPresets()).map(storedPresetFromResource);
      setStoredPresets(next);
      await refreshSessions();
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function loadArtifact(name: string) {
    if (busy) return;
    setBusy(true);
    try {
      const accepted = await api.loadModel(name, contextSize, 2048, {
        pin: loadPinned,
        idle_ttl_seconds: loadIdleTtl,
      });
      setSelectedJobId(accepted.operation_id);
      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      setModel(name);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function finishModelRegistration(names: string[]) {
    const nextArtifacts = await api.modelArtifacts(true);
    setArtifacts(nextArtifacts);
    const registered = nextArtifacts.filter((item) => names.includes(item.name));
    if (!registered.length) {
      throw new Error(tr("所选目录中的模型没有出现在模型目录中。", "Models from the selected folder were not registered in the catalog."));
    }
    if (registered.length === 1) {
      const artifact = registered[0];
      if (!artifact.loadable) {
        throw new Error(artifact.error || tr("所选模型不完整或无法加载。", "The selected model is incomplete or cannot be loaded."));
      }
      const loaded = instances.some((item) => item.model === artifact.name && item.state !== "failed")
        || runtime?.model === artifact.name;
      if (!loaded) {
        const accepted = await api.loadModel(artifact.name, contextSize, 2048, {
          pin: loadPinned,
          idle_ttl_seconds: loadIdleTtl,
        });
        setSelectedJobId(accepted.operation_id);
      }
    }
    setModelBrowserOpen(false);
    navigate(STUDIO_PATHS.models);
    await refreshRuntime(false);
    if (registered.length === 1) setModel(registered[0].name);
  }

  async function openModelDirectory(directoryId?: string | null, path?: string | null) {
    if (busy) return;
    setBusy(true);
    try {
      const listing = await api.modelDirectories(directoryId, path);
      setModelBrowser(listing);
      setModelDirectoryPath(listing.current_path ?? "");
      setModelBrowserOpen(true);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function jumpToModelDirectory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = modelDirectoryPath.trim();
    if (!path) return;
    await openModelDirectory(null, path);
  }

  async function chooseModelDirectory() {
    if (busy) return;
    if (!canUseNativeModelPicker) {
      await openModelDirectory();
      return;
    }
    setBusy(true);
    try {
      const names = await selectLocalModelDirectory();
      if (names) await finishModelRegistration(names);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function registerCurrentModelDirectory() {
    if (busy || !modelBrowser?.current_id) return;
    setBusy(true);
    try {
      const registered = await api.registerModelDirectory(modelBrowser.current_id);
      await finishModelRegistration(registered.map((item) => item.name));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function saveRuntimeProfile() {
    const name = profileName.replace(/\s+/g, " ").trim();
    const artifact = artifacts.find((item) => item.name === runtime?.model);
    if (busy || !name || !artifact) return;
    setBusy(true);
    try {
      await api.createRuntimeProfile({
        name,
        load: {
          model: artifact.name,
          device_ids: [],
          idle_ttl_seconds: loadPinned ? null : loadIdleTtl,
          pin: loadPinned,
          context_size: contextSize,
          prefill_chunk_size: 2048,
          sampling_defaults: {
            max_tokens: resolvedGlobalSettings.maxTokens,
            temperature: resolvedGlobalSettings.temperature,
            top_k: resolvedGlobalSettings.topK,
            top_p: resolvedGlobalSettings.topP,
            presence_penalty: resolvedGlobalSettings.presencePenalty,
            frequency_penalty: resolvedGlobalSettings.frequencyPenalty,
            repetition_penalty: resolvedGlobalSettings.repetitionPenalty,
            seed: resolvedGlobalSettings.seed,
            enable_thinking: thinkingSupported && resolvedGlobalSettings.enableThinking,
            enable_vision: resolvedGlobalSettings.enableVision,
            enable_mtp: resolvedGlobalSettings.enableMtp,
            reasoning_effort: resolvedGlobalSettings.reasoningEffort || null,
          },
        },
      });
      setProfileName("");
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function loadRuntimeProfile(profile: RuntimeProfile) {
    if (busy) return;
    if (profile.drifted && !await studioConfirm(tr(
      "模型产物已变化。仍使用这个配置档案加载？",
      "The model artifact changed. Load this profile anyway?",
    ))) return;
    setBusy(true);
    try {
      const accepted = await api.loadRuntimeProfile(profile.id, profile.drifted);
      setSelectedJobId(accepted.operation_id);
      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      setModel(profile.load.model);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function deleteRuntimeProfile(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      await api.deleteRuntimeProfile(id);
      setRuntimeProfiles((current) => current.filter((item) => item.id !== id));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function unloadInstance(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      const accepted = await api.unloadModel(id);
      setSelectedJobId(accepted.operation_id);
      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const last = runtime?.last_request;
  const lastPrefill = displayPrefillMetric(last);
  const lastTtftMs = preferPositiveMetric(last?.ttft_ms, last?.complete_prefill_ms);
  const runtimeModelName = runtime?.model || "Empty";
  const modelHero = useMemo(() => {
    const activeLoadJob = jobs.find(
      (job) => job.kind === "model.load"
        && ["queued", "running", "cancelling"].includes(job.status),
    );
    const latestLifecycleJob = jobs.find(
      (job) => job.kind === "model.load" || job.kind === "model.unload",
    );
    const currentInstance = instances.find((instance) => instance.id === runtime?.instance_id);
    const loadingInstance = [...instances].reverse().find(
      (instance) => instance.state === "loading",
    );
    const readyInstance = currentInstance?.state === "ready" || currentInstance?.state === "busy"
      ? currentInstance
      : [...instances].reverse().find(
        (instance) => instance.state === "ready" || instance.state === "busy",
      );
    const failedInstance = currentInstance?.state === "failed"
      ? currentInstance
      : [...instances].reverse().find((instance) => instance.state === "failed");
    const jobModel = typeof activeLoadJob?.payload.model === "string"
      ? activeLoadJob.payload.model
      : "";
    const runtimeState = String(runtime?.runtime_state || "").toLowerCase();

    if (runtime?.reloading || runtimeState === "loading" || loadingInstance || activeLoadJob) {
      return {
        name: loadingInstance?.model || jobModel || runtimeModelName,
        state: "loading" as const,
      };
    }
    if (runtimeState === "failed" || currentInstance?.state === "failed") {
      return {
        name: currentInstance?.model || runtimeModelName,
        state: "failed" as const,
      };
    }
    if (runtime?.model || readyInstance) {
      return {
        name: runtime?.model || readyInstance?.model || runtimeModelName,
        state: "ready" as const,
      };
    }
    if (failedInstance) {
      return {
        name: failedInstance.model || runtimeModelName,
        state: "failed" as const,
      };
    }
    if (
      latestLifecycleJob?.kind === "model.load"
      && latestLifecycleJob.status === "failed"
    ) {
      const failedJobModel = typeof latestLifecycleJob.payload.model === "string"
        ? latestLifecycleJob.payload.model
        : "";
      return {
        name: failedJobModel || runtimeModelName,
        state: "failed" as const,
      };
    }
    return { name: runtimeModelName, state: "idle" as const };
  }, [instances, jobs, runtime, runtimeModelName]);
  const modelHeroStatus = modelHero.state === "loading"
    ? tr("加载中", "Loading")
    : modelHero.state === "ready"
      ? tr("运行中", "Running")
      : modelHero.state === "failed"
        ? tr("加载失败", "Failed")
        : tr("空闲", "Idle");
  const activeJobs = jobs.filter((job) =>
    ["queued", "running", "cancelling"].includes(job.status),
  );
  const voiceComponentJob = activeJobs.find(
    (job) => job.kind === "component.voice_output.install",
  );
  const interactionMode = active?.mode ?? mode;
  const needsVoiceOutputComponent =
    interactionMode === "full_duplex" &&
    Boolean(capabilities?.model_capabilities.features.audio_output) &&
    !realtimeAvailable &&
    voiceComponent !== null;
  const completedJobs = jobs.filter(isTerminalJob);
  const runtimeMemory = Number(
    runtime?.mlx_active_bytes ??
      runtime?.cuda_allocated_bytes ??
      runtime?.process_resident_bytes ??
      0,
  );
  const runtimeCache = Number(runtime?.mlx_cache_bytes ?? runtime?.cuda_reserved_bytes ?? 0);
  const runtimeMemoryCapacity = Number(runtime?.device_total_bytes || 0)
    || Math.max(runtimeMemory + runtimeCache, 1);
  const runtimeDeviceFree = Number(runtime?.device_free_bytes || 0);
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheSnapshots = Number(runtime?.prefix_cache_snapshots || 0);
  const prefixCacheBytes = Number(runtime?.prefix_cache_bytes || 0);
  const prefixCacheDiskBytes = Number(runtime?.prefix_cache_disk_bytes || 0);
  const prefixCacheDiskBudget = Number(runtime?.prefix_cache_disk_max_bytes || 0);
  const prefixCacheHotBytes = Number(runtime?.prefix_cache_hot_bytes ?? prefixCacheBytes);
  const prefixCacheHitRate = prefixCacheQueries > 0
    ? (prefixCacheHits / prefixCacheQueries) * 100
    : 0;
  const prefixCacheHotOnly = runtime?.prefix_cache_mode === "single_device_hot_prefix";
  const prefixCachePersistent = typeof runtime?.prefix_cache_max_bytes === "number";
  const prefixCacheSupported = prefixCachePersistent || prefixCacheHotOnly;
  const genericJobKinds = jobKinds;
  const selectedKind = genericJobKinds.find((item) => item.kind === selectedJobKind);
  const imatrixArtifacts = lineage.filter((item) =>
    item.producer_kind === "calibrate.imatrix"
      || item.producer_kind === "artifact.import"
      || item.metadata?.media_type === "application/x-mfq-imatrix"
      || item.artifact_name.endsWith(".imatrix")
  );
  const selectedJob = jobs.find((item) => item.id === selectedJobId) ?? null;
  const clusterPanel = (
    <TMPanel className="cluster-panel">
      <div className="panel-heading">
        <div><h2>{tr("远程节点", "Remote nodes")}</h2><p>{tr("按模型和负载路由到健康的 MFQ Server", "Route by model and load across healthy MFQ Server nodes")}</p></div>
        <b>{remoteNodes.filter((node) => node.healthy).length} / {remoteNodes.length}</b>
      </div>
      <form className="node-form" onSubmit={registerRemoteNode}>
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("节点名称", "Node name")} value={nodeDraft.name} />
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, url: event.target.value }))} placeholder="https://worker.example" type="url" value={nodeDraft.url} />
        <input onChange={(event) => setNodeDraft((current) => ({ ...current, api_key_env: event.target.value }))} placeholder={tr("密钥环境变量（可选）", "Credential environment variable (optional)")} value={nodeDraft.api_key_env} />
        <button disabled={busy} type="submit">{tr("添加", "Add")}</button>
      </form>
      <div className="node-list">{remoteNodes.map((node) => <div key={node.id}><span className={node.healthy ? "model-state active" : "model-state failed"} /><div><strong>{node.name}</strong><small>{node.url} · {tr(`${node.models.length} 个模型`, `${node.models.length} models`)} · {tr(`${node.active_requests} 个活动请求`, `${node.active_requests} active`)}{typeof node.metrics.total_requests === "number" ? tr(` · ${formatNumber(node.metrics.total_requests)} 个请求`, ` · ${formatNumber(node.metrics.total_requests)} requests`) : ""}{node.error ? ` · ${node.error}` : ""}</small></div><button onClick={() => void api.deleteRemoteNode(node.id).then(() => setRemoteNodes((current) => current.filter((item) => item.id !== node.id))).catch((cause) => setError(errorMessage(cause)))} type="button">{tr("删除", "Delete")}</button></div>)}</div>
    </TMPanel>
  );

  const toolsRoutingPanel = <>
    <SectionLabel title={tr("工具与路由", "Tools and routing")} subtitle={tr("可选的 MCP 与远程节点", "Optional MCP and remote nodes")} />
    <div className="dashboard-grid server-tools-grid">
      <TMPanel className="mcp-panel">
        <div className="panel-heading"><div><h2>MCP</h2><p>{tr("工具服务器与模型可见工具", "Tool servers and model-visible tools")}</p></div><b>{tr(`${mcpTools.length} 个工具`, `${mcpTools.length} tools`)}</b></div>
        <form className="mcp-form" onSubmit={createMcpServer}><input aria-label={tr("服务器名称", "Server name")} onChange={(event) => setMcpDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("名称", "Name")} value={mcpDraft.name} /><select aria-label={tr("传输方式", "Transport")} onChange={(event) => setMcpDraft((current) => ({ ...current, transport: event.target.value as "stdio" | "streamable_http" }))} value={mcpDraft.transport}><option value="streamable_http">HTTP</option><option value="stdio">stdio</option></select><input aria-label={mcpDraft.transport === "streamable_http" ? "Streamable HTTP URL" : tr("可执行文件", "Executable")} onChange={(event) => setMcpDraft((current) => ({ ...current, endpoint: event.target.value }))} placeholder={mcpDraft.transport === "streamable_http" ? "https://host/mcp" : tr("可执行文件路径", "Executable path")} type={mcpDraft.transport === "streamable_http" ? "url" : "text"} value={mcpDraft.endpoint} /><button className="primary" disabled={busy || !mcpDraft.name.trim() || !mcpDraft.endpoint.trim()} type="submit">{tr("添加", "Add")}</button></form>
        <div className="mcp-server-list">{mcpServers.map((server) => <div className="mcp-server" key={server.id}><span className={server.enabled ? "model-state active" : "model-state"} /><div><strong>{server.name}</strong><small>{server.transport} · {server.url || server.command}</small></div><button onClick={() => void toggleMcpServer(server)} type="button">{server.enabled ? tr("停用", "Disable") : tr("启用", "Enable")}</button><button onClick={() => void deleteMcpServer(server.id)} type="button">{tr("删除", "Delete")}</button></div>)}</div>
      </TMPanel>
      {clusterPanel}
    </div>
  </>;

  const serverDraft = studioDraft ?? studio?.config ?? null;
  const serverActive = Boolean(studio?.reachable);
  const serverPage = (
    <div className="server-page">
      {serverActive && <div className="server-active-notice">
        <Icon name="info" size={15} />
        <span>{tr("服务器正在运行；网络设置保存后会立即重新连接。", "The server is active. Network changes reconnect as soon as they are saved.")}</span>
      </div>}

      <SectionLabel title={tr("运行服务", "Runtime")} />
      <TMPanel className="server-settings-panel">
        <div className="setting-list">
          <SettingRow
            title={tr("Runtime 可执行文件", "Runtime executable")}
            detail={tr("应用已包含推理服务，并自动使用本机 Metal Runtime。", "The packaged app includes the inference server and discovers the local Metal runtime automatically.")}
            trailing={<div className="server-row-actions"><code>mfq-cli → mfq-decode-metal</code><span className={`runtime-status-pill ${serverActive ? "running" : "stopped"}`}><i />{serverActive ? tr("已连接", "Connected") : tr("离线", "Offline")}</span></div>}
          />
          <SettingRow
            title={tr("模型 ID", "Model ID")}
            detail={tr("由 /v1/models 公布，并用于对话补全请求。", "Advertised by /v1/models and accepted by chat completions.")}
            trailing={<div className="server-row-actions server-model-control">{availableModelNames.length > 1 ? <select aria-label={tr("当前模型", "Current model")} disabled={busy || sessionTransitioning} onChange={(event) => selectModel(event.target.value)} value={model}>{availableModelNames.map((name) => <option key={name} value={name}>{name}</option>)}</select> : <strong title={model}>{model || tr("尚未加载", "Not loaded")}</strong>}<button onClick={() => openStudioPage("dashboard", "models")} type="button">{tr("选择…", "Choose…")}</button></div>}
          />
          <SettingRow
            title={tr("绑定地址", "Bind address")}
            detail={tr("本地模式仅监听 127.0.0.1；远程模式连接另一台 MFQ Server。", "Local mode stays on 127.0.0.1; remote mode connects to another MFQ Server.")}
            trailing={<select aria-label={tr("绑定地址", "Bind address")} disabled={busy || !serverDraft} onChange={(event) => setStudioDraft((current) => current && ({ ...current, mode: event.target.value as StudioConfig["mode"] }))} value={serverDraft?.mode ?? "local"}><option value="local">{tr("仅本机 · 127.0.0.1", "Local only · 127.0.0.1")}</option><option value="remote">{tr("远程 MFQ Server", "Remote MFQ Server")}</option></select>}
          />
          {serverDraft?.mode === "remote" ? <>
            <SettingRow
              title={tr("远程端点", "Remote endpoint")}
              detail={tr("远程 MFQ Server 的 OpenAI 兼容基础 URL。", "OpenAI-compatible base URL for the remote MFQ Server.")}
              trailing={<input aria-label={tr("远程端点", "Remote endpoint")} className="server-wide-input" disabled={busy} onChange={(event) => setStudioDraft((current) => current && ({ ...current, remote_url: event.target.value }))} placeholder="https://host:port" type="url" value={serverDraft.remote_url} />}
            />
            <SettingRow
              title={tr("API 密钥", "API key")}
              detail={tr("凭据只保存在系统凭据库中。", "The credential is stored only in the system credential vault.")}
              trailing={<input aria-label={tr("API 密钥", "API key")} autoComplete="off" className="server-wide-input" disabled={busy} onChange={(event) => { setStudioToken(event.target.value); setStudioCredentialWritable(true); }} placeholder={tr("可选", "Optional")} type="password" value={studioToken} />}
            />
          </> : <SettingRow
            title={tr("端口", "Port")}
            detail={tr("OpenAI 兼容 HTTP 服务使用的 TCP 端口。", "TCP port used by the OpenAI-compatible HTTP server.")}
            trailing={<input aria-label={tr("端口", "Port")} className="server-number-input" disabled={busy || !serverDraft} max={65535} min={1} onChange={(event) => setStudioDraft((current) => current && ({ ...current, local_service_port: Number(event.target.value) }))} type="number" value={serverDraft?.local_service_port ?? 8090} />}
          />}
        </div>
      </TMPanel>

      <SectionLabel title={tr("内存规划", "Memory plan")} />
      <TMPanel className="server-settings-panel">
        <div className="setting-list">
          <SettingRow
            title={tr("模型总驻留", "Total model residency")}
            detail={tr("MFQ 根据模型、专家缓存和设备可用内存自动规划驻留。", "MFQ plans model, expert-cache, and device residency from the available memory automatically.")}
            trailing={<strong>{runtimeMemory + runtimeCache > 0 ? formatBytes(runtimeMemory + runtimeCache) : tr("自动", "Automatic")}</strong>}
          />
          <SettingRow
            title={tr("前缀 RAM 配额", "Prefix RAM allowance")}
            detail={tr("RAM 热层从统一内存预算中分配，剩余空间可用于模型和专家缓存。", "The RAM hot tier is allocated inside the unified budget, leaving the remainder for model and expert caching.")}
            trailing={<strong>{prefixCacheHotBytes > 0 ? formatBytes(prefixCacheHotBytes) : tr("自动", "Automatic")}</strong>}
          />
          <SettingRow
            title={tr("启动时预热专家缓存", "Warm expert cache on launch")}
            detail={tr("MFQ 会依据架构和当前内存压力自动预热可用专家槽位。", "MFQ warms available expert slots according to the architecture and current memory pressure.")}
            trailing={<span className="server-managed-value">{tr("自动", "Automatic")}</span>}
          />
          <SettingRow
            title={tr("前缀块大小", "Prefix block size")}
            detail={tr("较大的块有利于长提示词吞吐，较小的块提供更细粒度的部分前缀复用。", "Larger blocks favor long-prompt throughput; smaller blocks allow finer partial-prefix reuse.")}
            trailing={<div className="server-unit-value"><strong>{formatNumber(runtime?.prefix_cache_block_tokens || 256)}</strong><span>tokens</span></div>}
          />
          <SettingRow
            title={tr("最大上下文", "Maximum context")}
            detail={tr("修改后重载当前模型；未加载模型时会作为下一次加载的默认值。", "Reloads the active model after a change; otherwise it becomes the next load default.")}
            trailing={<div className="server-row-actions"><div className="server-input-unit"><input aria-label={tr("最大上下文", "Maximum context")} className="server-number-input" max={1048576} min={512} onChange={(event) => setContextSize(Number(event.target.value))} type="number" value={contextSize} /><span>tokens</span></div><button disabled={busy || !runtime?.model} onClick={() => void reloadRuntime()} type="button">{tr("重载", "Reload")}</button></div>}
          />
        </div>
      </TMPanel>

      <SectionLabel title={tr("持久化前缀缓存", "Persistent Prefix cache")} />
      <TMPanel className="server-settings-panel">
        <div className="setting-list">
          <SettingRow
            title={tr("启用 SSD 层", "Enable SSD tier")}
            detail={tr("保存经过校验的增量前缀块，并允许服务重启后继续复用。", "Persist incremental, checksummed Prefix blocks and recover them after restart.")}
            trailing={<input aria-label={tr("启用 SSD 层", "Enable SSD tier")} checked={prefixCachePersistent && !prefixCacheHotOnly} disabled readOnly type="checkbox" />}
          />
          <SettingRow
            title={tr("目录", "Directory")}
            detail={tr("MFQ 自动选择应用数据目录中的本地缓存位置。", "MFQ automatically uses a local cache location inside the application data directory.")}
            trailing={<code>{tr("由 MFQ 管理", "Managed by MFQ")}</code>}
          />
          <SettingRow
            title={tr("SSD 配额", "SSD budget")}
            detail={tr("基于 LRU 的回收策略会将磁盘占用控制在此上限内。", "Leaf-aware LRU keeps disk usage within this ceiling.")}
            trailing={<div className="server-unit-value"><strong>{prefixCacheDiskBudget > 0 ? formatNumber(prefixCacheDiskBudget / 2 ** 30, 0) : "--"}</strong><span>GiB</span></div>}
          />
        </div>
      </TMPanel>

      <SectionLabel title={tr("对话", "Chat")} />
      <TMPanel className="server-settings-panel">
        <div className="setting-list">
          <SettingRow
            title={tr("系统提示词", "System prompt")}
            detail={tr("添加到内置对话中新请求的开头。", "Prepended to new requests in the built-in playground.")}
            trailing={<textarea aria-label={tr("系统提示词", "System prompt")} className="server-prompt-input" onChange={(event) => updateGlobalInference({ systemPrompt: event.target.value })} placeholder={tr("系统提示词", "System prompt")} rows={2} value={settings.systemPrompt} />}
          />
          <SettingRow
            title={tr("最大输出", "Maximum output")}
            detail={tr("内置对话的最大生成长度。", "Completion limit for the built-in chat.")}
            trailing={<div className="server-input-unit"><input aria-label={tr("最大输出", "Maximum output")} className="server-number-input" max={65536} min={1} onChange={(event) => updateGlobalInference({ maxTokens: Number(event.target.value) })} type="number" value={settings.maxTokens} /><span>tokens</span></div>}
          />
          <SettingRow
            title={tr("温度", "Temperature")}
            detail={tr("采样温度范围为 0 到 2。", "Sampling temperature between 0 and 2.")}
            trailing={<div className="server-temperature-control"><input aria-label={tr("温度", "Temperature")} max={2} min={0} onChange={(event) => updateGlobalInference({ temperature: Number(event.target.value) })} step={0.05} type="range" value={settings.temperature} /><strong>{formatNumber(settings.temperature, 2)}</strong></div>}
          />
        </div>
      </TMPanel>

      <SectionLabel title={tr("自动化", "Automation")} />
      <TMPanel className="server-settings-panel">
        <div className="setting-list">
          <SettingRow
            title={tr("MFQ Studio 打开时启动服务器", "Start server when MFQ Studio opens")}
            detail={tr("本地模式会自动恢复服务，并使用当前模型和已保存的运行配置。", "Local mode restores the server automatically with the current model and saved runtime configuration.")}
            trailing={<input aria-label={tr("MFQ Studio 打开时启动服务器", "Start server when MFQ Studio opens")} checked={serverDraft?.mode !== "remote"} disabled readOnly type="checkbox" />}
          />
        </div>
      </TMPanel>

      <div className="server-page-footer">
        <span>{tr("对话默认值会自动保存。", "Chat defaults are saved automatically.")}</span>
        <button className="primary" disabled={busy || sessionTransitioning || !serverDraft} onClick={() => void saveStudioSettings()} type="button">{tr("保存服务器设置", "Save server settings")}</button>
      </div>
    </div>
  );



  const resetSettingsDraft = () => setSettingsDraft({
    ...modeTemplateSettings({
      ...DEFAULT_SETTINGS,
      language: settingsDraft.language,
      theme: settingsDraft.theme,
      playbackEnabled: settingsDraft.playbackEnabled,
    }, active?.mode ?? mode, runtime, realtime),
    inheritModelDefaults: true,
  });

  const settingsPage = <SettingsPage
    tr={tr}
    settingsDraft={settingsDraft}
    setSettingsDraft={setSettingsDraft}
    mtpAvailable={mtpAvailable}
    presetManager={presetManager}
    contextCapacity={runtime?.context_capacity}
    contextSize={contextSize}
    setContextSize={setContextSize}
    busy={busy}
    hasStudio={Boolean(studio)}
    actions={{ setModelDefaultInheritance, applyPreset, reloadRuntime, exportStudioData, importStudioData, openServerPage, resetSettingsDraft, saveSettings }}
  />;

  const dashboardCopy: Record<DashboardPage, { title: string; subtitle: string }> = {
    overview: {
      title: tr("概览", "Overview"),
      subtitle: tr("启动服务、查看内存层级并复制 API 端点。", "Start the server, watch the memory hierarchy, and copy the API endpoint."),
    },
    models: {
      title: tr("模型", "Models"),
      subtitle: tr("管理本地检查点和已加载实例。", "Manage local checkpoints and loaded instances."),
    },
    connections: {
      title: tr("服务器", "Server"),
      subtitle: tr("配置原生 Runtime、内存规划、持久化前缀缓存与内置对话默认值。", "Configure the native runtime, memory plan, persistent Prefix cache, and built-in chat defaults."),
    },
    cache: {
      title: tr("资源与缓存", "Resources"),
      subtitle: tr("检查内存层级、前缀缓存与运行配置。", "Inspect memory, prefix caching, and runtime profiles."),
    },
    logs: {
      title: tr("日志", "Logs"),
      subtitle: tr("查看请求、后台任务和服务事件。", "Inspect requests, background jobs, and server events."),
    },
    settings: {
      title: tr("设置", "Settings"),
      subtitle: tr("配置界面与全局推理默认值。", "Configure the interface and global inference defaults."),
    },
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#studio-main">{tr("跳到主要内容", "Skip to main content")}</a>
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`} id="studio-sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div><strong>MFQ</strong><span>Studio</span></div>
        </div>
        <div className="sidebar-scroll">
          <nav className="sectioned-nav" aria-label={tr("推理", "Inference")}>
            <section>
              <div className="sidebar-group-label">{tr("推理", "Inference")}</div>
              <button aria-current={view === "dashboard" && dashboardPage === "overview" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "overview" ? "active" : ""} onClick={() => openStudioPage("dashboard", "overview")} type="button"><Icon name="gauge" />{tr("概览", "Overview")}{Number(runtime?.active_requests || 0) > 0 && <span>{formatNumber(runtime?.active_requests || 0)}</span>}</button>
              <button aria-current={view === "dashboard" && dashboardPage === "models" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "models" ? "active" : ""} onClick={() => openStudioPage("dashboard", "models")} type="button"><Icon name="folder" />{tr("模型", "Models")}</button>
              <button aria-current={view === "dashboard" && dashboardPage === "connections" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "connections" ? "active" : ""} onClick={openServerPage} type="button"><Icon name="server-rack" />{tr("服务器", "Server")}</button>
              <button aria-current={view === "dashboard" && dashboardPage === "cache" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "cache" ? "active" : ""} onClick={() => openStudioPage("dashboard", "cache")} type="button"><Icon name="memory" />{tr("资源", "Resources")}</button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr("交互", "Playground")}</div>
              <button aria-current={view === "chat" ? "page" : undefined} className={view === "chat" ? "active" : ""} onClick={openChatPage} type="button"><Icon name="chat" />{tr("对话", "Chat")}</button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr("模型工具", "Model tools")}</div>
              <button className={view === "lab" && labPage === "models" ? "active" : ""} onClick={() => openStudioPage("lab", "models")} type="button"><Icon name="download" />{tr("模型仓库", "Model hub")}</button>
              <button className={view === "lab" && labPage === "evaluations" ? "active" : ""} onClick={() => openStudioPage("lab", "evaluations")} type="button"><Icon name="activity" />{tr("评测与数据集", "Evaluations")}</button>
              <button className={view === "lab" && labPage === "quantization" ? "active" : ""} onClick={() => openStudioPage("lab", "quantization")} type="button"><Icon name="memory" />{tr("量化工作台", "Quantization")}</button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr("系统", "System")}</div>
              <button aria-current={view === "dashboard" && dashboardPage === "logs" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "logs" ? "active" : ""} onClick={() => openStudioPage("dashboard", "logs")} type="button"><Icon name="activity" />{tr("日志", "Logs")}</button>
              <button aria-current={view === "dashboard" && dashboardPage === "settings" ? "page" : undefined} className={view === "dashboard" && dashboardPage === "settings" ? "active" : ""} onClick={openSettings} type="button"><Icon name="settings" />{tr("设置", "Settings")}</button>
            </section>
          </nav>
        </div>
        <button className="sidebar-runtime-card" onClick={() => openStudioPage("dashboard", "overview")} type="button">
          <span className={`runtime-dot ${Number(runtime?.active_requests || 0) > 0 ? "busy" : selectedModelAvailable ? "ready" : selectedModelLoading ? "busy" : "idle"}`} />
          <span>
            <strong>{model || tr("服务空闲", "Server idle")}</strong>
            <small>{availableModelNames.length > 1 ? tr(`${availableModelNames.length} 个模型已加载`, `${availableModelNames.length} models loaded`) : selectedModelAvailable ? `${formatNumber(runtime?.active_requests || 0)} ${tr("个活动请求", "active requests")}` : selectedModelLoading ? tr("模型加载中", "Model loading") : tr("选择模型以开始", "Choose a model to begin")}</small>
          </span>
          <Icon name="activity" size={14} />
        </button>
      </aside>
      <button
        aria-controls="studio-sidebar"
        aria-expanded={sidebarOpen}
        aria-label={tr("打开侧栏", "Open sidebar")}
        className="mobile-menu-trigger"
        onClick={openSidebar}
        type="button"
      >
        <Icon name="menu" size={17} />
      </button>
      <button aria-label={tr("关闭侧栏", "Close sidebar")} className={`mobile-scrim ${sidebarOpen ? "open" : ""}`} onClick={closeSidebar} type="button" />

      <main className="workspace" id="studio-main">
        {view === "chat" ? (
          <section className={"chat-view " + (chatSessionsOpen ? "sessions-open" : "sessions-collapsed")}>
            <aside className={"chat-session-sidebar" + (chatSessionsOpen ? ' open' : '')} aria-label={tr('会话列表', 'Conversations')}>
              <div className="chat-session-sidebar-header"><strong>{tr('对话', 'Chats')}</strong><button aria-label={tr('新建会话', 'New chat')} className="chat-icon-button" disabled={busy || sessionTransitioning || !selectedModelAvailable} onClick={() => void createSession()} title={tr('新建会话', 'New chat')} type="button"><Icon name="plus" size={15} /></button></div>
              <div className="chat-session-list">{sessions.length ? sessions.map((session) => <button aria-current={session.id === activeId ? 'page' : undefined} className={session.id === activeId ? 'active' : ''} disabled={busy || sessionTransitioning} key={session.id} onClick={() => selectSession(session.id)} title={session.title || tr('未命名会话', 'Untitled chat')} type="button"><strong>{session.title || tr('未命名会话', 'Untitled chat')}</strong><small>{session.model}</small></button>) : <p>{tr('暂无会话', 'No conversations yet')}</p>}</div>
            </aside>
            <header className="chat-screen-header">
              <div className="chat-screen-title"><button aria-expanded={chatSessionsOpen} aria-label={chatSessionsOpen ? tr('收起会话列表', 'Collapse conversations') : tr('展开会话列表', 'Expand conversations')} className="chat-sidebar-toggle" onClick={() => setChatSessionsOpen((open) => !open)} title={chatSessionsOpen ? tr('收起会话列表', 'Collapse conversations') : tr('展开会话列表', 'Expand conversations')} type="button"><span aria-hidden="true">{chatSessionsOpen ? '‹' : '›'}</span></button><h1>{active?.title || tr('对话', 'Chat')}</h1></div>
              <div className="chat-screen-actions"><div className="chat-model-summary">{availableModelNames.length > 1 ? <select aria-label={tr('对话模型', 'Chat model')} disabled={busy || sessionTransitioning} onChange={(event) => selectModel(event.target.value)} value={model}>{availableModelNames.map((name) => <option key={name} value={name}>{name}</option>)}</select> : <strong>{model || tr('尚未加载模型', 'No model loaded')}</strong>}<small>{tr('最多 ' + formatNumber(effectiveSettings.maxTokens) + ' tokens', formatNumber(effectiveSettings.maxTokens) + ' max tokens')} · {tr('温度', 'temperature')} {formatNumber(effectiveSettings.temperature, 2)} · {tr('流式', 'streaming')}</small></div><span className={'runtime-status-pill ' + (conversationReady ? 'running' : 'stopped')}><i />{conversationReady ? tr('就绪', 'Ready') : selectedModelLoading ? tr('加载中', 'Loading') : tr('空闲', 'Idle')}</span><button aria-label={tr('清空对话', 'Clear conversation')} className="chat-icon-button" disabled={!conversationReady || busy || (!messages.length && !currentVoiceMessages.length)} onClick={() => void clearActiveConversation()} title={tr('清空对话', 'Clear conversation')} type="button"><Icon name="trash" size={14} /></button></div>
            </header>
            <div className="message-scroller" onScroll={handleMessageScroll} ref={messageScrollerRef}>
              <div className="message-list" aria-live="polite">
                {!messages.length && !currentVoiceMessages.length && !live && <div className="welcome"><Icon name="chat" size={34} />{!selectedModelAvailable ? <><h1>{selectedModelLoading ? tr("模型加载中", "Model loading") : tr("尚未加载模型", "No model loaded")}</h1><p>{selectedModelLoading ? tr("加载完成后即可开始对话。", "Chat becomes available as soon as loading completes.") : tr("选择本地检查点后即可开始对话。", "Choose a local checkpoint to use the inference playground.")}</p>{!selectedModelLoading && <button className="open-model-primary" disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="folder" />{tr("选择模型", "Choose model")}</button>}</> : !active ? <><h1>{tr("本机私密对话", "A private conversation on your Mac")}</h1><p>{tr("请求直接发送到本机 MFQ 服务，不经过云端中转。", "Requests go directly to the local MFQ Runtime with no cloud relay.")}</p><button className="open-model-primary" disabled={busy} onClick={() => void createSession()} type="button">{tr("开始对话", "Start chat")}</button></> : <><h1>{tr("本机私密对话", "A private conversation on your Mac")}</h1><p>{tr("请求直接发送到本机 MFQ 服务，不经过云端中转。", "Requests go directly to the local MFQ Runtime with no cloud relay.")}</p></>}</div>}
                <SavedMessageList messages={messages} responses={responses} mcpTools={mcpTools} busy={busy} tr={tr} editDraft={editDraft} setEditDraft={setEditDraft} actions={{ saveEdit, copyMessage, regenerate, executeToolCalls }} />
                {currentVoiceMessages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className="message-body">{message.text && renderMarkdown(message.text, false, message.role === "assistant")}{message.audioId && <AudioClip audioId={message.audioId} />}</div></article>)}
                {liveVoice?.sessionId === activeId && liveVoice.text && <article className="message message-assistant live-message"><div className="message-body">{renderMarkdown(liveVoice.text, true, true)}</div></article>}
                {live && <article className="message message-assistant live-message"><div className="message-body">{live.reasoning && <details className="reasoning" open><summary>{tr("正在思考", "Thinking")}</summary>{renderMarkdown(live.reasoning, true, true)}</details>}{live.text && renderMarkdown(live.text, true, true)}{live.tools.map((tool, index) => <pre className="tool-call" key={index}>{tool}</pre>)}{!live.reasoning && !live.text && live.tools.length === 0 && <span className="thinking"><i /><i /><i /></span>}</div></article>}
              </div>
            </div>
            {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError(null)} type="button">×</button></div>}
            <div className="composer-region">
              {needsVoiceOutputComponent && voiceComponent && <div className="voice-component-banner"><div><strong>{voiceComponent.ready ? tr("语音组件已下载", "Voice component downloaded") : tr("此模型还缺少语音输出组件", "This model needs the voice output component")}</strong><span>{voiceComponentJob ? tr(`正在下载并校验 · ${formatNumber(voiceComponentJob.progress * 100)}%`, `Downloading and verifying · ${formatNumber(voiceComponentJob.progress * 100)}%`) : voiceComponent.error ? voiceComponent.error : tr("Token2Wav 独立安装，不会重复占用每个模型的空间。", "Token2Wav is installed once and shared by all compatible models.")}</span></div>{voiceComponentJob && <progress max={1} value={voiceComponentJob.progress} />}<button disabled={voiceComponentBusy || Boolean(voiceComponentJob)} onClick={() => void installOrEnableVoiceOutput()} type="button">{voiceComponentJob ? tr("正在下载…", "Downloading…") : voiceComponent.ready ? tr("启用语音输出", "Enable voice output") : tr(`下载组件 · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`, `Download · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`)}</button></div>}
              <form className="composer" onSubmit={send}>
                {attachments.length > 0 && <div className="attachment-tray">{attachments.map((attachment) => <div className="attachment-chip" key={attachment.id}>{attachment.kind === "image" ? <img alt="" src={attachment.previewUrl} /> : attachment.kind === "video" ? <VideoWithFirstFrame muted src={attachment.previewUrl} /> : <span>{attachment.kind === "document" ? "TXT" : "♫"}</span>}<div><strong>{attachment.file.name}</strong><small>{attachment.kind} · {formatNumber(attachment.file.size)} B</small></div><button aria-label={tr("移除附件", "Remove attachment")} onClick={() => removeAttachment(attachment.id)} type="button">×</button></div>)}</div>}
                <textarea aria-label={tr("消息", "Message")} disabled={!conversationReady || busy} maxLength={32768} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder={conversationReady ? tr("向模型发送消息", "Message MFQ") : selectedModelAvailable ? tr("正在切换对话模型", "Switching chat model") : tr("请先加载模型", "Load a model first")} rows={1} value={draft} />
                <div className="composer-toolbar">
                  <input accept={attachmentAccept} hidden multiple onChange={(event) => selectAttachments(event.target.files)} ref={attachmentInputRef} type="file" />
                  <button aria-label={tr("添加附件", "Add attachment")} disabled={!conversationReady || busy} onClick={() => attachmentInputRef.current?.click()} title={tr("添加文档或媒体", "Add document or media")} type="button"><Icon name="paperclip" /></button>
                  {capabilities && (capabilities.model_capabilities.features.audio_input || capabilities.model_capabilities.features.full_duplex) && <select aria-label={tr("交互模式", "Interaction mode")} disabled={!conversationReady || busy || voiceState !== "idle"} onChange={(event) => void selectInteractionMode(event.target.value as SessionMode)} value={active?.mode ?? mode}>{(["text", "voice", "full_duplex"] as SessionMode[]).map((item) => { const feature = capabilities.model_capabilities.features; const disabled = item === "voice" ? !feature.audio_input : item === "full_duplex" ? !feature.full_duplex : false; return <option disabled={disabled} key={item} value={item}>{MODE_LABELS[item][english ? 1 : 0]}</option>; })}</select>}
                  {realtimeAvailable && <button aria-label={tr("语音输入", "Voice input")} aria-pressed={voiceState !== "idle" && voiceState !== "error"} className="voice-button" disabled={!conversationReady || active?.mode === "text" || busy} onClick={() => void toggleVoice()} style={{ "--voice-level": voiceLevel } as React.CSSProperties} title={active?.mode === "text" ? tr("请先选择语音或全双工模式", "Select voice or full duplex mode first") : voiceState === "processing" ? tr("语音处理中", "Processing voice") : tr("语音输入", "Voice input")} type="button"><span /></button>}
                  {realtimeAvailable && active?.mode !== "text" && <button aria-label={tr("语音播放", "Voice playback")} aria-pressed={settings.playbackEnabled} onClick={() => setSettings((current) => ({ ...current, playbackEnabled: !current.playbackEnabled }))} title={tr("语音播放", "Voice playback")} type="button"><Icon name={settings.playbackEnabled ? "volume" : "volume-off"} /></button>}
                  {active?.mode === "text" && visionSupported && <button aria-label={tr("视觉输入", "Vision input")} aria-pressed={visionAvailable && effectiveSettings.enableVision} disabled={!visionAvailable} onClick={() => updateGlobalInference({ enableVision: !effectiveSettings.enableVision })} title={visionAvailable ? tr("视觉输入", "Vision input") : tr("当前模型文件没有视觉权重", "The current model artifact has no vision weights")} type="button"><Icon name="image" />{tr("视觉", "Vision")}</button>}
                  {active?.mode === "text" && mtpSupported && <button aria-label="MTP" aria-pressed={mtpAvailable && effectiveSettings.enableMtp} disabled={!mtpAvailable} onClick={() => updateGlobalInference({ enableMtp: !effectiveSettings.enableMtp })} title={mtpAvailable ? "MTP" : tr("当前模型文件没有完整 MTP 权重", "The current model artifact has no complete MTP head")} type="button"><Icon name="text-forward" />MTP</button>}
                  {active?.mode === "text" && <button aria-pressed={thinkingSupported && effectiveSettings.enableThinking} disabled={!thinkingSupported} onClick={() => updateGlobalInference({ enableThinking: !effectiveSettings.enableThinking })} type="button"><Icon name="lightbulb" />{tr("思考", "Thinking")}</button>}
                  {active?.mode === "text" && thinkingSupported && effectiveSettings.enableThinking && reasoningValues.length > 0 && <select aria-label={tr("思考档位", "Reasoning effort")} onChange={(event) => updateGlobalInference({ reasoningEffort: event.target.value })} value={effectiveSettings.reasoningEffort}><option value="">{tr("标准", "Standard")}</option>{reasoningValues.map((value) => <option key={value} value={value}>{value}</option>)}</select>}
                  <span className="composer-hint">{voiceState !== "idle" ? voiceState : tr("Enter 发送 · Shift+Enter 换行", "Enter to send · Shift+Enter for newline")}</span>
                  {busy ? <button aria-label={stopping ? tr("正在停止生成", "Stopping generation") : tr("停止生成", "Stop generation")} className="send-button stop" disabled={stopping} onClick={() => void stopGeneration()} type="button"><Icon name="stop" size={14} /></button> : <button aria-label={tr("发送", "Send")} className="send-button" disabled={!conversationReady || (!draft.trim() && !attachments.length)} type="submit"><Icon name="send" size={15} /></button>}
                </div>
              </form>
              <p>{tr("模型输出可能存在错误，请核对重要信息。", "Model output may be inaccurate. Verify important information.")}</p>
            </div>
          </section>
        ) : view === "dashboard" ? (
          <section className="dashboard-view" id="dashboard-overview">
            <ScreenHeader
              title={dashboardCopy[dashboardPage].title}
              subtitle={dashboardCopy[dashboardPage].subtitle}
              trailing={dashboardPage === "overview" ? <button disabled={busy} onClick={() => void refreshRuntime()} type="button"><Icon name="refresh" size={14} />{tr("刷新", "Refresh")}</button> : dashboardPage === "models" ? <button disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="plus" size={14} />{tr("添加模型", "Add model")}</button> : dashboardPage === "settings" ? <button className="primary" onClick={saveSettings} type="button">{tr("应用设置", "Apply settings")}</button> : undefined}
            />
            {dashboardPage === "overview" && <TMPanel className="runtime-hero">
              <ModelMonogram name={modelHero.name} state={modelHero.state} />
              <div className="runtime-hero-copy">
                <div><h2>{modelHero.name}</h2><span className={`runtime-status-pill ${modelHero.state}`}><i />{modelHeroStatus}</span></div>
                <p className="runtime-endpoint">{studio?.service_url || "http://127.0.0.1:8090"}</p>
                <small>{runtime?.model ? `${runtime?.model_type || "MFQ"} · ${formatNumber(runtime?.max_context)} ${tr("上下文", "context")} · ${formatDuration(runtime?.uptime_seconds)}` : tr("加载本地模型后即可开始推理。", "Load a local model to begin inference.")}</small>
              </div>
              <div className="runtime-hero-actions">
                <button onClick={() => openStudioPage("dashboard", "models")} type="button"><Icon name="folder" size={15} />{tr("模型", "Models")}</button>
                <button className="primary" onClick={openChatPage} type="button"><Icon name="chat" size={15} />{tr("对话", "Chat")}</button>
              </div>
            </TMPanel>}
            {dashboardPage === "overview" && <>
              {availableModelNames.length > 1 && <>
                <SectionLabel title={tr("已加载模型", "Loaded models")} subtitle={tr(`${availableModelNames.length} 个可用于推理`, `${availableModelNames.length} available for inference`)} />
                <TMPanel className="overview-models-panel">
                  <div className="overview-model-grid">{availableModelNames.map((name) => {
                    const instance = instances.find((candidate) => candidate.model === name && candidate.state !== "failed");
                    const selected = name === model;
                    const stateLabel = instance?.state === "busy" ? tr("使用中", "Busy") : tr("就绪", "Ready");
                    const details = [
                      instance?.devices.join(" + "),
                      instance?.context_size ? `${formatNumber(instance.context_size)} ctx` : null,
                      instance ? tr(`${instance.active_sessions} 个会话`, `${instance.active_sessions} sessions`) : null,
                    ].filter(Boolean).join(" · ");
                    return <button aria-pressed={selected} className={`overview-model-card${selected ? " selected" : ""}`} disabled={busy || sessionTransitioning || selected} key={name} onClick={() => selectModel(name)} type="button">
                      <ModelMonogram name={name} state="ready" />
                      <span className="overview-model-copy"><strong title={name}>{name}</strong><small>{details || stateLabel}</small></span>
                      <span className="runtime-status-pill ready"><i />{selected ? tr("当前", "Current") : stateLabel}</span>
                    </button>;
                  })}</div>
                </TMPanel>
              </>}
              <SectionLabel title={tr("实时性能", "Live performance")} subtitle={tr("最近请求吞吐与累计缓存复用", "Latest request throughput · cumulative cache reuse")} />
              <div className="metric-grid">
                <MetricTile label={tr("预填充", "Prefill")} value={`${formatNumber(lastPrefill.tokensPerSecond, 1)} tok/s`} detail={`${formatNumber(lastPrefill.milliseconds, 1)} ms · ${tr("输入处理", "Prompt processing")}`} icon="text-forward" />
                <MetricTile label={tr("解码", "Decode")} value={`${formatNumber(last?.decode_tps, 1)} tok/s`} detail={tr("输出生成", "Token generation")} icon="waveform" />
                <MetricTile label={tr("首字延迟", "TTFT")} value={`${formatNumber(lastTtftMs, 1)} ms`} detail={tr("首次输出耗时", "Time to first token")} icon="clock" />
                <MetricTile label={tr("前缀复用", "Prefix reuse")} value={prefixCacheQueries > 0 ? `${formatNumber(prefixCacheHitRate, 1)}%` : "--"} detail={tr(`已恢复 ${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens`, `${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens restored`)} icon="reuse" />
                <MetricTile label={tr("内存", "Memory")} value={formatBytes(runtimeMemory)} detail={runtimeCache ? tr(`分配器缓存 ${formatBytes(runtimeCache)}`, `${formatBytes(runtimeCache)} allocator cache`) : tr("模型驻留", "Runtime residency")} icon="memory" />
              </div>
              <SectionLabel title={tr("内存层级", "Memory hierarchy")} />
              {runtime ? <TMPanel className="overview-memory-panel">
                <div className="overview-memory-heading">
                  <div><h2>{tr("推理内存", "Runtime memory")}</h2><p>{tr("模型驻留、分配器缓存与可复用前缀共享统一内存。", "Model residency, allocator cache, and reusable prefixes share unified memory.")}</p></div>
                  <strong>{formatBytes(runtimeMemory + runtimeCache)}</strong>
                </div>
                <div className="overview-memory-bars">
                  <UsageBar label={tr("模型与活动张量", "Model and active tensors")} used={runtimeMemory} total={runtimeMemoryCapacity} />
                  <UsageBar label={tr("分配器缓存", "Allocator cache")} used={runtimeCache} total={runtimeMemoryCapacity} />
                </div>
                <div className="overview-memory-facts">
                  <span><Icon name="memory" size={13} />{formatBytes(runtimeDeviceFree)} {tr("设备可用", "device free")}</span>
                  <span><Icon name="reuse" size={13} />{formatNumber(runtime?.prefix_cache_snapshots || 0)} {tr("个前缀快照", "prefix snapshots")}</span>
                  <span><Icon name="text-forward" size={13} />{tr(`已复用 ${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens`, `${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens reused`)}</span>
                  <span><Icon name="queue" size={13} />{formatNumber(runtime?.active_requests || 0)} {tr("个活动请求", "active requests")}</span>
                </div>
              </TMPanel> : <EmptyPanel icon="memory" title={tr("推理内存尚未上报", "Runtime memory is unavailable")} message={tr("服务器就绪后会显示模型驻留与缓存状态。", "Memory residency and cache state appear when the server is ready.")} />}
              <div className="overview-footer-grid">
                <TMPanel className="overview-endpoint-panel">
                  <div className="overview-panel-title"><Icon name="link" size={15} /><h2>{tr("OpenAI 兼容端点", "OpenAI-compatible endpoint")}</h2><button aria-label={endpointCopied ? tr("已复制", "Copied") : tr("复制端点", "Copy endpoint")} className={endpointCopied ? "copied" : ""} onClick={() => void copyEndpoint()} title={endpointCopied ? tr("已复制", "Copied") : tr("复制端点", "Copy endpoint")} type="button"><Icon name={endpointCopied ? "check" : "copy"} size={14} /></button></div>
                  <code>{studio?.service_url || "http://127.0.0.1:8090"}</code>
                  <p>{tr("可直接用于 OpenAI SDK；默认回环地址不经过云端。", "Use this base URL with OpenAI SDKs. The default loopback address sends no traffic to the cloud.")}</p>
                </TMPanel>
                <TMPanel className="overview-session-panel">
                  <div className="overview-panel-title"><Icon name="chart" size={15} /><h2>{tr("会话统计", "Session")}</h2><span>{formatDuration(runtime?.uptime_seconds)}</span></div>
                  <div className="overview-session-stats">
                    <div><strong>{formatNumber(runtime?.total_requests || 0)}</strong><small>{tr("已完成", "Completed")}</small></div>
                    <div><strong>{formatNumber(runtime?.total_prompt_tokens || 0)}</strong><small>{tr("提示词", "Prompt")}</small></div>
                    <div><strong>{formatNumber(runtime?.total_completion_tokens || 0)}</strong><small>{tr("已生成", "Generated")}</small></div>
                  </div>
                  <p>{formatNumber(runtime?.failed_requests || 0)} {tr("个失败请求", "failed requests")} · {formatNumber(runtime?.active_requests || 0)} {tr("个活动请求", "active")}</p>
                </TMPanel>
              </div>
            </>}
            {dashboardPage === "settings" && settingsPage}
            {dashboardPage === "cache" && <>
              <SectionLabel title={tr("内存层级", "Memory hierarchy")} subtitle={tr("设备、内存与持久缓存", "Device, memory, and persistent cache")} />
              {prefixCacheSupported ? (
                <TMPanel className="cache-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Session KV cache</h2>
                      <p>{prefixCacheHotOnly
                        ? tr("当前进程内的单条设备热前缀；切换会话或编辑 prompt 会重新 prefill", "One device-hot prefix in this process; switching sessions or editing the prompt triggers a fresh prefill")
                        : tr("RAM 热缓存与可跨重启复用的 SSD 前缀块", "RAM hot cache with persistent SSD prefix blocks")}</p>
                    </div>
                    <b>{prefixCacheQueries > 0 ? `${formatNumber(prefixCacheHitRate, 1)}% hit` : tr("暂无查询", "No queries")}</b>
                  </div>
                  {prefixCacheHotOnly ? (
                    <>
                    <UsageBar label={tr("RAM 热前缀", "RAM hot prefix")} used={prefixCacheHotBytes} total={Math.max(prefixCacheHotBytes, runtimeCache || 0)} />
                    <div className="cache-stats">
                      <div><span>{tr("活动会话", "Active sessions")}</span><strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong></div>
                      <div><span>{tr("设备热前缀", "Device-hot prefix")}</span><strong>{formatNumber(prefixCacheSnapshots)}</strong><small>{tr("单物理快照", "one physical snapshot")}</small></div>
                      <div><span>{tr("缓存 tokens", "Cached tokens")}</span><strong>{formatNumber(runtime?.prefix_cache_tokens)}</strong></div>
                      <div><span>{tr("复用 tokens", "Reused tokens")}</span><strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong></div>
                      <div><span>{tr("命中 / 查询", "Hits / queries")}</span><strong>{formatNumber(prefixCacheHits)} / {formatNumber(prefixCacheQueries)}</strong></div>
                      <div><span>{tr("持久性", "Persistence")}</span><strong>{tr("进程生命周期", "Process lifetime")}</strong><small>{tr("不落盘", "not stored on disk")}</small></div>
                    </div>
                    </>
                  ) : (
                    <>
                    <div className="cache-usage-bars">
                      <UsageBar label={tr("SSD 前缀缓存", "SSD prefix cache")} used={prefixCacheDiskBytes} total={Math.max(prefixCacheDiskBudget, prefixCacheDiskBytes)} />
                      <UsageBar label={tr("RAM 热层", "RAM hot tier")} used={prefixCacheHotBytes} total={Math.max(prefixCacheHotBytes, runtimeCache || 0)} />
                    </div>
                    <div className="cache-stats">
                      <div><span>{tr("活动会话", "Active sessions")}</span><strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong></div>
                      <div><span>{tr("SSD 块", "SSD blocks")}</span><strong>{formatNumber(runtime?.prefix_cache_disk_blocks ?? prefixCacheSnapshots)}</strong></div>
                      <div><span>{tr("复用 tokens", "Reused tokens")}</span><strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong></div>
                      <div><span>{tr("SSD 占用", "SSD usage")}</span><strong>{formatNumber(prefixCacheDiskBytes / 2 ** 30, 2)} GB</strong><small>{prefixCacheDiskBudget > 0 ? `${formatNumber(prefixCacheDiskBudget / 2 ** 30, 0)} GB ${tr("上限", "limit")}` : ""}</small></div>
                      <div><span>{tr("RAM 热层", "RAM hot tier")}</span><strong>{formatNumber(prefixCacheHotBytes / 2 ** 20, 1)} MB</strong><small>{formatNumber(runtime?.prefix_cache_hot_blocks)} blocks</small></div>
                      <div><span>{tr("待写入", "Pending writes")}</span><strong>{formatNumber(runtime?.prefix_cache_pending_writes)}</strong><small>{formatNumber(Number(runtime?.prefix_cache_pending_bytes || 0) / 2 ** 20, 1)} / {formatNumber(Number(runtime?.prefix_cache_pending_max_bytes || 0) / 2 ** 20, 0)} MB · {formatNumber(runtime?.prefix_cache_deduplicated_writes)} deduplicated</small></div>
                      <div><span>{tr("SSD 命中", "SSD hits")}</span><strong>{formatNumber(runtime?.prefix_cache_disk_hits)}</strong><small>{formatNumber(runtime?.prefix_cache_hot_hits)} RAM hits</small></div>
                      <div><span>{tr("回收", "Evictions")}</span><strong>{formatNumber(runtime?.prefix_cache_evictions)}</strong><small>{formatNumber(runtime?.prefix_cache_corrupt_blocks)} corrupt</small></div>
                    </div>
                    </>
                  )}
                  {prefixCacheSnapshots > 0 && <button className="panel-action danger" disabled={busy || Number(runtime?.active_requests || 0) > 0} onClick={() => void clearRuntimeCache()} type="button">{tr("清除 Session KV 缓存", "Clear Session KV cache")}</button>}
                </TMPanel>
              ) : <EmptyPanel icon="settings" title={tr("Runtime 诊断未启用", "Runtime diagnostics are offline")} message={tr("加载模型后可查看内存与前缀缓存状态。", "Load a model to inspect memory and prefix-cache state.")} />}
              <SectionLabel title={tr("运行配置", "Runtime profiles")} />
              <TMPanel className="profile-panel"><div className="panel-heading"><div><h2>{tr("已保存配置", "Saved profiles")}</h2><p>{tr("将加载参数和采样默认值绑定到模型产物", "Bind load and sampling defaults to a model artifact")}</p></div><b>{runtimeProfiles.length}</b></div><div className="profile-create"><input maxLength={64} onChange={(event) => setProfileName(event.target.value)} placeholder={tr("当前配置名称", "Current configuration name")} value={profileName} /><button disabled={busy || !profileName.trim() || !artifacts.some((item) => item.name === runtime?.model)} onClick={() => void saveRuntimeProfile()} type="button">{tr("保存当前配置", "Save current")}</button></div>{runtimeProfiles.length > 0 ? <div className="profile-list">{runtimeProfiles.map((profile) => <div className={`profile-row ${profile.drifted ? "drifted" : ""}`} key={profile.id}><div><strong>{profile.name}</strong><small>{profile.load.context_size.toLocaleString()} ctx · {profile.load.prefill_chunk_size.toLocaleString()} chunk{profile.drifted ? ` · ${tr("模型已变化", "artifact changed")}` : ""}</small></div><button disabled={busy} onClick={() => void loadRuntimeProfile(profile)} type="button">{tr("加载", "Load")}</button><button aria-label={tr("删除配置档案", "Delete profile")} disabled={busy} onClick={() => void deleteRuntimeProfile(profile.id)} type="button"><Icon name="trash" size={14} /></button></div>)}</div> : <div className="inline-empty">{tr("尚未保存运行配置。", "No runtime profiles saved yet.")}</div>}</TMPanel>
              {toolsRoutingPanel}
            </>}
            {dashboardPage === "models" && <>
              <div className="model-workbench-summary">
                <div><span>{tr("运行中的模型", "Loaded models")}</span><strong>{availableModelNames.length}</strong><small>{tr("可直接用于对话", "Ready for chat")}</small></div>
                <div><span>{tr("本地检查点", "Local checkpoints")}</span><strong>{artifacts.length}</strong><small>{tr("已登记到 MFQ", "Registered in MFQ")}</small></div>
                <div><span>{tr("当前对话模型", "Chat model")}</span><strong title={model || undefined}>{model || tr("未选择", "None")}</strong><small>{model ? tr("切换会话模型不会重新注册资产", "Switching keeps the registered asset") : tr("加载后从这里选择", "Choose one after loading")}</small></div>
                <div className="model-workbench-links"><button onClick={() => openStudioPage("lab", "models")} type="button"><Icon name="download" size={13} />{tr("打开模型仓库", "Open model hub")}</button><button onClick={() => openStudioPage("lab", "quantization")} type="button"><Icon name="memory" size={13} />{tr("去量化", "Quantize")}</button></div>
              </div>
              <div className="model-catalog-toolbar"><div><strong>{tr("模型资产", "Model assets")}</strong><span>{tr("注册、加载和切换对话模型", "Register, load, and switch chat models")}</span></div><label><span aria-hidden="true">/</span><input aria-label={tr("筛选模型", "Filter models")} onChange={(event) => setModelFilter(event.target.value)} placeholder={tr("按名称筛选", "Filter by name")} value={modelFilter} /></label></div>
              <SectionLabel title={tr("已加载模型", "Loaded models")} subtitle={tr(`${availableModelNames.length} 个可用于推理`, `${availableModelNames.length} available for inference`)} />
              {filteredInstances.length > 0 ? <TMPanel className="model-catalog-panel loaded-model-panel"><div className="model-list">{filteredInstances.map((instance) => {
                const ready = instance.state === "ready" || instance.state === "busy";
                const selected = instance.model === model;
                const stateLabel = instance.state === "loading" ? tr("加载中", "Loading") : instance.state === "unloading" ? tr("卸载中", "Unloading") : instance.state === "failed" ? tr("失败", "Failed") : instance.state === "busy" ? tr("使用中", "Busy") : tr("就绪", "Ready");
                return <div className="model-row" key={instance.id}><span className={instance.state === "failed" ? "model-state failed" : ready ? "model-state active" : "model-state"} /><div><strong>{instance.model}</strong><small>{stateLabel} · {formatNumber(instance.context_size)} ctx{instance.pinned ? ` · ${tr("固定", "Pinned")}` : instance.idle_ttl_seconds != null ? ` · TTL ${instance.idle_ttl_seconds}s` : ""}</small></div><div className="model-row-actions">{ready && <button className={selected ? "selected" : ""} disabled={busy || sessionTransitioning || selected} onClick={() => selectModel(instance.model)} type="button">{selected ? tr("当前", "Current") : tr("用于对话", "Use in chat")}</button>}<button disabled={busy || instance.state !== "ready"} onClick={() => void unloadInstance(instance.id)} type="button">{tr("卸载", "Unload")}</button></div></div>;
              })}</div></TMPanel> : <EmptyPanel icon="memory" title={tr(modelFilter ? "没有匹配的已加载模型" : "当前没有已加载模型", modelFilter ? "No loaded model matches" : "No models loaded")} message={tr("从本地检查点加载一个模型后即可开始对话。", "Load a local checkpoint to start chatting.")} action={<button className="primary" disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="folder" size={14} />{tr("添加模型", "Add model")}</button>} />}
              <TMPanel className="model-catalog-panel">
                <div className="panel-heading"><div><h2>{tr("加载策略", "Load policy")}</h2><p>{tr("控制模型的驻留与自动卸载。", "Control model residency and automatic unloading.")}</p></div></div>
                <div className="setting-list model-policy-panel">
                  <SettingRow title={tr("固定到内存", "Pin in memory")} detail={tr("跳过 LRU 与空闲卸载", "Skip LRU and idle eviction")} trailing={<input aria-label={tr("固定到内存", "Pin in memory")} checked={loadPinned} onChange={(event) => setLoadPinned(event.target.checked)} type="checkbox" />} />
                  <SettingRow title={tr("空闲卸载", "Idle unload")} detail={loadPinned ? tr("固定模型不使用 TTL", "Ignored while pinned") : tr("每次使用后重新计时", "Resets after each use")} trailing={<select disabled={loadPinned} onChange={(event) => setLoadIdleTtl(event.target.value ? Number(event.target.value) : null)} value={loadIdleTtl ?? ""}><option value="">{tr("永不", "Never")}</option><option value="300">5 min</option><option value="900">15 min</option><option value="3600">1 h</option></select>} />
                </div>
              </TMPanel>
              <SectionLabel title={tr("本地检查点", "Local checkpoints")} subtitle={`${artifacts.length} ${tr("个本地模型", "local models")}`} />
              {filteredArtifacts.length > 0 ? <TMPanel className="model-catalog-panel model-library-panel"><div className="model-list">{filteredArtifacts.map((item) => {
                  const instance = instances.find((candidate) => candidate.model === item.name && candidate.state !== "failed");
                  const loaded = Boolean(instance) || item.name === runtime?.model;
                  const policy = instance?.pinned ? tr("固定", "Pinned") : instance?.idle_ttl_seconds != null ? `TTL ${instance.idle_ttl_seconds}s` : null;
                  return <div className="model-row" key={item.id}><span className={loaded ? "model-state active" : item.loadable ? "model-state" : "model-state failed"} /><div><strong>{item.name}</strong><small>{item.architecture} · {item.shard_count} shards · {formatNumber(item.total_bytes / 2 ** 30, 1)} GB{policy ? ` · ${policy}` : ""}</small></div>{instance ? <button disabled={busy || instance.state !== "ready"} onClick={() => void unloadInstance(instance.id)} type="button">{tr("卸载", "Unload")}</button> : loaded ? <em>{tr("已加载", "Loaded")}</em> : !item.loadable ? <em className="failed" title={item.error || undefined}>{item.complete && item.format === "hf" ? tr("需先转换", "Convert first") : tr("不可用", "Invalid")}</em> : <button disabled={busy} onClick={() => void loadArtifact(item.name)} type="button">{tr("加载", "Load")}</button>}</div>;
                })}</div></TMPanel> : <EmptyPanel icon="folder" title={tr(modelFilter ? "没有匹配的本地模型" : "还没有本地模型", modelFilter ? "No local model matches" : "No local models yet")} message={tr("添加一个模型文件夹即可开始。", "Add a model folder to get started.")} action={<button className="primary" disabled={busy} onClick={() => void chooseModelDirectory()} type="button"><Icon name="folder" size={14} />{tr("选择模型文件夹", "Choose model folder")}</button>} />}
            </>}
            {dashboardPage === "logs" && <>
              <SectionLabel title={tr("Runtime 活动", "Runtime activity")} subtitle={tr("请求、任务与事件", "Requests, jobs, and events")} />
              <div className="dashboard-grid logs-grid">
              <TMPanel><div className="panel-heading"><div><h2>{tr("最近请求", "Recent requests")}</h2></div></div>{requestHistory.length > 0 ? <div className="request-table">{requestHistory.slice(0, 8).map((request) => <div className="request-row" key={request.id}><div><strong>{request.id}</strong><small>{request.completed_at ? new Date(request.completed_at * 1000).toLocaleTimeString() : request.endpoint || "completion"}</small></div><span>{formatNumber(request.prompt_tokens)} → {formatNumber(request.completion_tokens)}</span><b>{formatNumber(request.decode_tps, 1)} tok/s</b></div>)}</div> : <div className="inline-empty">{tr("还没有完成的请求。", "No completed requests yet.")}</div>}</TMPanel>
              <TMPanel>
                <div className="panel-heading"><div><h2>{tr("后台任务", "Background jobs")}</h2></div><b>{activeJobs.length}</b></div>
                {activeJobs.length > 0 && <div className="request-table">{activeJobs.slice(0, 8).map((job) => <div className="job-row" key={job.id}><div><strong>{job.kind}</strong><small>{new Date(job.updated_at).toLocaleTimeString()} · {job.status}</small></div><progress max={1} value={job.progress} /><b>{formatNumber(job.progress * 100)}%</b></div>)}</div>}
                {completedJobs.length > 0 && <details className="completed-jobs">
                  <summary><span>{tr("已完成", "Completed")} <b>{completedJobs.length}</b></span><button disabled={jobCleanupBusy} onClick={(event) => { event.preventDefault(); void clearCompletedJobRecords(); }} type="button">{tr("清理已完成", "Clear completed")}</button></summary>
                  <div className="request-table">{completedJobs.slice(0, 8).map((job) => <div className="job-row completed" key={job.id}><div><strong>{job.kind}</strong><small>{new Date(job.updated_at).toLocaleTimeString()} · {job.status}</small></div><progress max={1} value={job.progress} /><b>{formatNumber(job.progress * 100)}%</b><button aria-label={tr("移出任务历史", "Remove from job history")} disabled={jobCleanupBusy} onClick={() => void deleteJobRecord(job.id)} type="button"><Icon name="trash" size={12} /></button></div>)}</div>
                </details>}
              </TMPanel>
              <TMPanel className="runtime-log-panel"><div className="panel-heading"><div><h2>{tr("Runtime 日志", "Runtime logs")}</h2></div><b>{runtimeLogs.length}</b></div>{runtimeLogs.length > 0 ? <div className="runtime-log-list">{runtimeLogs.slice(-8).reverse().map((entry) => <div className={`runtime-log ${entry.level}`} key={entry.sequence}><span>{new Date(entry.created_at).toLocaleTimeString()}</span><p>{entry.message}</p></div>)}</div> : <div className="inline-empty">{tr("暂无 Runtime 事件。", "No runtime events yet.")}</div>}</TMPanel>
              </div>
            </>}
            {dashboardPage === "connections" && serverPage}
          </section>
        ) : (
          <section aria-label={tr("模型工具", "Model tools")} className="lab-view">
            <div className="page-heading"><div><h1>{labPage === "models" ? tr("模型仓库", "Model hubs") : labPage === "evaluations" ? tr("评测与数据集", "Evaluations") : tr("量化工作台", "Quantization workspace")}</h1><p>{labPage === "models" ? tr("浏览模型来源并登记本地资产。", "Browse model sources and register local assets.") : labPage === "evaluations" ? tr("组织数据集、评测结果与可复现评测流程。", "Organize datasets, results, and reproducible evaluation workflows.") : tr("配置并运行 MFQ 量化流程。", "Configure and run MFQ quantization workflows.")}</p></div></div>
            <nav aria-label={tr("模型工具页面", "Model tools pages")} className="workspace-tabs">
              <button aria-current={labPage === "models" ? "page" : undefined} className={labPage === "models" ? "active" : ""} onClick={() => openStudioPage("lab", "models")} type="button"><Icon name="download" size={14} />{tr("模型仓库", "Model hub")}</button>
              <button aria-current={labPage === "evaluations" ? "page" : undefined} className={labPage === "evaluations" ? "active" : ""} onClick={() => openStudioPage("lab", "evaluations")} type="button"><Icon name="activity" size={14} />{tr("评测与数据集", "Evaluations")}</button>
              <button aria-current={labPage === "quantization" ? "page" : undefined} className={labPage === "quantization" ? "active" : ""} onClick={() => openStudioPage("lab", "quantization")} type="button"><Icon name="memory" size={14} />{tr("量化工作台", "Quantization")}</button>
            </nav>
            {labPage === "models" && <PanelDeck labels={panelLabels} page="lab-models"><div key="hubs"><section className="dashboard-panel hub-panel"><div className="panel-heading"><div><h2>{tr("模型仓库", "Model hubs")}</h2><p>{tr("搜索、粘贴仓库链接并启动可续传下载", "Search or paste a repository link to start a resumable download")}</p></div></div><form aria-busy={busy} className="hub-search" onSubmit={searchHub}><select onChange={(event) => setHubProvider(event.target.value as HubModelSummary["provider"])} value={hubProvider}><option value="modelscope">ModelScope</option><option value="huggingface">Hugging Face</option></select><input onChange={(event) => setHubQuery(event.target.value)} placeholder={tr("模型名称、仓库或链接", "Model, repository, or URL")} value={hubQuery} /><button disabled={busy || !hubQuery.trim()} type="submit">{busy ? tr("搜索中…", "Searching…") : tr("查找", "Find")}</button></form>{hubQuery.trim() && hubResults.length === 0 && !hubModel && <div className="inline-empty hub-empty">{tr("没有匹配的模型仓库。检查名称、组织名或仓库链接。", "No model hub matches. Check the model name, organization, or repository URL.")}</div>}{hubResults.length > 0 && <div className="hub-results">{hubResults.map((item) => <button className={hubModel?.repo_id === item.repo_id ? "active" : ""} key={`${item.provider}:${item.repo_id}`} onClick={() => void inspectHubModel(item)} type="button"><div><strong>{item.repo_id}</strong><small>{formatNumber(item.downloads)} downloads · {formatNumber(item.likes)} likes</small></div><span>{item.total_bytes ? `${formatNumber(item.total_bytes / 2 **30, 1)} GB` : "--"}</span></button>)}</div>}{hubModel && <div className="hub-detail"><div><strong>{hubModel.repo_id}</strong><small>{hubModel.revision} · {hubModel.files.length} files · {formatNumber(hubModel.total_bytes / 2 ** 30, 2)} GB</small></div><button disabled={busy || !jobKinds.some((item) => item.kind === `download.${hubModel.provider}`)} onClick={() => void downloadHubModel()} type="button"><Icon name="download" size={14} />{tr("下载", "Download")}</button></div>}</section></div></PanelDeck>}
            {labPage === "evaluations" && <PanelDeck labels={panelLabels} page="lab-evaluations">
              <section className="dashboard-panel evaluation-panel" key="results">
                <div className="panel-heading"><div><h2>{tr("评测结果", "Evaluation results")}</h2><p>{tr("只允许数据集与运行参数一致的结果对比", "Comparison requires matching datasets and execution parameters")}</p></div><b>{evaluations.length}</b></div>
                <div className="evaluation-list">{evaluations.length === 0 ? <div className="inline-empty">{tr("还没有评测结果。先注册数据集并运行评测任务。", "No evaluation results yet. Register a dataset and run an evaluation job first.")}</div> : evaluations.map((item) => <label key={item.id}><input checked={selectedEvaluations.includes(item.id)} onChange={(event) => setSelectedEvaluations((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} type="checkbox" /><div><strong>{item.model_id}</strong><small>{item.kind} · {new Date(item.created_at).toLocaleString()}</small></div><span>{Object.entries(item.metrics).filter(([, value]) => typeof value === "number").slice(0, 2).map(([name, value]) => `${name} ${formatNumber(Number(value), 3)}`).join(" · ")}</span></label>)}</div>
                <button className="panel-action" disabled={busy || selectedEvaluations.length < 2} onClick={() => void compareSelectedEvaluations()} type="button">{tr("对比所选结果", "Compare selected")}</button>
                {evaluationComparison && <div className="comparison-table"><header><span>{tr("模型", "Model")}</span>{evaluationComparison.metrics.map((metric) => <b key={metric}>{metric}</b>)}</header>{evaluationComparison.rows.map((row) => <div key={row.evaluation.id}><strong>{row.evaluation.model_id}</strong>{evaluationComparison.metrics.map((metric) => <span key={metric}>{formatNumber(Number(row.evaluation.metrics[metric]), 4)}<small>{row.deltas[metric] == null ? "" : ` ${Number(row.deltas[metric]) >= 0 ? "+" : ""}${formatNumber(Number(row.deltas[metric]), 4)}`}</small></span>)}</div>)}</div>}
              </section>
              <section className="dashboard-panel dataset-panel" key="datasets">
                <div className="panel-heading"><div><h2>{tr("数据集", "Datasets")}</h2><p>{tr("可复现的文件哈希与来源清单", "Reproducible file hashes and source manifests")}</p></div><b>{datasets.length}</b></div>
                <form className="dataset-form" onSubmit={registerDataset}><input onChange={(event) => setDatasetDraft((current) => ({ ...current, name: event.target.value }))} placeholder={tr("名称", "Name")} value={datasetDraft.name} /><select onChange={(event) => setDatasetDraft((current) => ({ ...current, kind: event.target.value as DatasetResource["kind"] }))} value={datasetDraft.kind}><option value="custom">Custom</option><option value="wikitext2">WikiText-2</option></select><input onChange={(event) => setDatasetDraft((current) => ({ ...current, artifact_uri: event.target.value }))} placeholder="workspace://datasets/corpus.txt" value={datasetDraft.artifact_uri} /><button disabled={busy} type="submit">{tr("注册", "Register")}</button></form>
                <div className="dataset-list">{datasets.length === 0 ? <div className="inline-empty">{tr("还没有数据集。注册一个文件或工作区资源后即可开始评测。", "No datasets yet. Register a file or workspace resource to start evaluating.")}</div> : datasets.map((item) => <div key={item.id}><div><strong>{item.name}</strong><small>{item.kind} · {formatNumber(item.byte_size / 2 ** 20, 2)} MiB · {item.sha256.slice(0, 12)}</small></div><button aria-label={tr("删除数据集", "Delete dataset")} onClick={() => void api.deleteDataset(item.id).then(() => setDatasets((current) => current.filter((entry) => entry.id !== item.id))).catch((cause) => setError(errorMessage(cause)))} type="button"><Icon name="trash" size={13} /></button></div>)}</div>
              </section>
            </PanelDeck>}
            {labPage === "quantization" && <PanelDeck labels={panelLabels} page="lab-quantization">
            <section className="dashboard-panel imatrix-panel" key="imatrix">
              <div className="panel-heading"><div><h2>Imatrix</h2><p>{tr("单独校准、导入，或在量化任务中先校准再使用", "Calibrate separately, import one, or collect it before quantization")}</p></div><b>{imatrixArtifacts.length}</b></div>
              <div className="imatrix-actions">
                <button disabled={busy || !jobKinds.some((item) => item.kind === "calibrate.imatrix")} onClick={() => setSelectedJobKind("calibrate.imatrix")} type="button"><Icon name="plus" size={11} />{tr("新建校准", "New calibration")}</button>
                <button disabled={busy || imatrixImporting || !jobKinds.some((item) => item.kind === "artifact.import")} onClick={() => imatrixInputRef.current?.click()} type="button"><Icon name="upload" size={11} />{imatrixImporting ? tr("正在导入", "Importing") : tr("导入文件", "Import file")}</button>
                <input accept=".imatrix,.gguf,.dat,application/x-mfq-imatrix,application/octet-stream" hidden onChange={(event) => void importImatrix(event.target.files)} ref={imatrixInputRef} type="file" />
              </div>
              {imatrixArtifacts.length > 0 && <div className="imatrix-list">{imatrixArtifacts.slice(0, 12).map((item) => <button key={item.id} onClick={() => { setPendingImatrix(item.artifact_uri.replace(/^workspace:\/\//, "")); setSelectedJobKind("model.quantize"); }} type="button"><div><strong>{item.artifact_name}</strong><small>{item.artifact_uri}</small></div><span>{tr("用于量化", "Use")}</span></button>)}</div>}
            </section>
            <section className="dashboard-panel lineage-panel" key="lineage"><div className="panel-heading"><div><h2>{tr("产物谱系", "Artifact lineage")}</h2><p>{tr("源产物、生成任务、默认后参数和验证记录", "Sources, producing jobs, resolved parameters, and validations")}</p></div><b>{lineage.length}</b></div><div className="lineage-list">{lineage.length === 0 ? <div className="inline-empty">{tr("暂无产物谱系记录。运行量化或导入任务后会显示在这里。", "No artifact lineage yet. Run a quantization or import job to populate this view.")}</div> : lineage.slice(0, 20).map((item) => <details key={item.id}><summary><div><strong>{item.artifact_name}</strong><small>{item.producer_kind} · {new Date(item.created_at).toLocaleString()}</small></div><span>{item.validation_job_ids.length} checks</span></summary><dl><div><dt>URI</dt><dd>{item.artifact_uri}</dd></div><div><dt>{tr("源", "Sources")}</dt><dd>{item.source_uris.join(", ") || "--"}</dd></div></dl><pre>{JSON.stringify(item.parameters, null, 2)}</pre></details>)}</div></section>
              <form className="dashboard-panel job-builder" key="builder" onSubmit={submitJob}>
                <div className="panel-heading"><div><h2>{tr("新任务", "New job")}</h2></div></div>
                <label><span>{tr("任务类型", "Job type")}</span><select onChange={(event) => setSelectedJobKind(event.target.value)} value={selectedJobKind}>{genericJobKinds.map((item) => <option key={item.kind} value={item.kind}>{item.kind}</option>)}</select></label>
                {Object.entries(selectedKind?.payload_schema.properties ?? {}).map(([name, property]) => {
                  if (
                    selectedJobKind === "model.quantize"
                    && name.startsWith("imatrix_")
                    && name !== "imatrix"
                    && !Boolean(jobPayload.calibrate_imatrix)
                  ) return null;
                  if (
                    selectedJobKind === "model.quantize"
                    && name === "imatrix"
                    && Boolean(jobPayload.calibrate_imatrix)
                  ) return null;
                  const required = selectedKind?.payload_schema.required?.includes(name);
                  const type = schemaType(property);
                  const value = jobPayload[name];
                  const isImatrix = selectedJobKind === "model.quantize" && name === "imatrix";
                  return <label key={name}><span>{property.title || name}{required ? " *" : ""}</span>{isImatrix && imatrixArtifacts.length > 0 ? <select onChange={(event) => updateJobPayload(name, property, event.target.value)} value={String(value ?? "")}><option value="">{tr("不使用", "None")}</option>{imatrixArtifacts.map((item) => <option key={item.id} value={item.artifact_uri.replace(/^workspace:\/\//, "")}>{item.artifact_name}</option>)}</select> : property.enum ? <select onChange={(event) => updateJobPayload(name, property, event.target.value)} required={required} value={String(value ?? "")}>{!required && <option value="" />}{property.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select> : type === "boolean" ? <input checked={Boolean(value)} onChange={(event) => updateJobPayload(name, property, event.target.checked)} type="checkbox" /> : <input max={property.maximum} min={property.minimum} onChange={(event) => updateJobPayload(name, property, event.target.value)} required={required} type={type === "integer" || type === "number" ? "number" : "text"} value={Array.isArray(value) ? value.join(", ") : String(value ?? "")} />}{property.description && <small>{property.description}</small>}</label>;
                })}
                <div className="job-submit">
                  <button className="job-run" disabled={busy || !selectedJobKind} type="submit"><Icon name="play" size={12} />{tr("运行", "Run")}</button>
                </div>
              </form>
              <section className="dashboard-panel job-history" key="history">
                <div className="panel-heading"><div><h2>{tr("任务历史", "Job history")}</h2></div><b>{activeJobs.length}</b></div>
                {activeJobs.length > 0 && <div className="job-list">{activeJobs.map((job) => <button className={job.id === selectedJobId ? "active" : ""} key={job.id} onClick={() => setSelectedJobId(job.id)} type="button"><span className={`job-status ${job.status}`} /><div><strong>{job.kind}</strong><small>{job.status} · {new Date(job.updated_at).toLocaleString()}</small></div><b>{formatNumber(job.progress * 100)}%</b></button>)}</div>}
                {completedJobs.length > 0 && <details className="completed-jobs">
                  <summary><span>{tr("已完成", "Completed")} <b>{completedJobs.length}</b></span><button disabled={jobCleanupBusy} onClick={(event) => { event.preventDefault(); void clearCompletedJobRecords(); }} type="button">{tr("清理已完成", "Clear completed")}</button></summary>
                  <div className="job-list">{completedJobs.map((job) => <button className={job.id === selectedJobId ? "active" : ""} key={job.id} onClick={() => setSelectedJobId(job.id)} type="button"><span className={`job-status ${job.status}`} /><div><strong>{job.kind}</strong><small>{job.status} · {new Date(job.updated_at).toLocaleString()}</small></div><b>{formatNumber(job.progress * 100)}%</b></button>)}</div>
                </details>}
              </section>
              {selectedJob && <section className="dashboard-panel job-detail" key="detail"><div className="panel-heading"><div><h2>{selectedJob.kind}</h2><p>{selectedJob.id}</p></div><b>{selectedJob.status}</b></div><progress max={1} value={selectedJob.progress} /><div className="job-result-grid"><div><span>{tr("进度", "Progress")}</span><strong>{formatNumber(selectedJob.progress * 100)}%</strong></div><div><span>{tr("更新时间", "Updated")}</span><strong>{new Date(selectedJob.updated_at).toLocaleTimeString()}</strong></div></div><div className="job-actions">{["queued", "running", "cancelling"].includes(selectedJob.status) && <button className="secondary" disabled={selectedJob.status === "cancelling"} onClick={() => void cancelSelectedJob()} type="button">{tr("取消任务", "Cancel job")}</button>}{["failed", "cancelled", "interrupted"].includes(selectedJob.status) && <button className="secondary" disabled={busy} onClick={() => void retrySelectedJob()} type="button">{tr("重试", "Retry")}</button>}{isTerminalJob(selectedJob) && <button className="secondary" disabled={jobCleanupBusy} onClick={() => void deleteJobRecord(selectedJob.id)} type="button">{tr("移出任务历史", "Remove from history")}</button>}{String(selectedJob.result?.artifact || "").startsWith("workspace://") && <button className="secondary danger" disabled={busy} onClick={() => void removeSelectedArtifact()} type="button">{tr("删除本地产物", "Delete local artifact")}</button>}</div>{selectedJob.error && <div className="job-error">{selectedJob.error.message}</div>}{selectedJob.result && <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>}<div className="job-log"><header><span>{tr("事件与日志", "Events and logs")}</span></header>{jobLogs.map((entry) => <div className={entry.level} key={entry.sequence}><time>{new Date(entry.created_at).toLocaleTimeString()}</time><code>{entry.message}</code></div>)}</div></section>}
            </PanelDeck>}
          </section>
        )}
      </main>

      {modelBrowserOpen && modelBrowser && <div className="dialog-backdrop"><section className="studio-dialog model-browser-dialog"><header><div><h2>{tr("选择模型文件夹", "Choose model folder")}</h2><p>{tr("浏览 MFQ Server 所在设备上的文件夹。", "Browse folders on the MFQ Server host.")}</p></div><button onClick={() => setModelBrowserOpen(false)} type="button">×</button></header><form className="model-browser-location" onSubmit={jumpToModelDirectory}><button disabled={busy || !modelBrowser.current_id} onClick={() => void openModelDirectory(modelBrowser.parent_id)} type="button">{tr("上一级", "Up")}</button><input aria-label={tr("当前目录", "Current directory")} onChange={(event) => setModelDirectoryPath(event.target.value)} placeholder={tr("输入服务器上的完整目录", "Enter a full directory on the server")} spellCheck={false} value={modelDirectoryPath} /><button disabled={busy || !modelDirectoryPath.trim()} type="submit">{tr("前往", "Go")}</button>{modelBrowser.current_id && <span>{modelBrowser.model_file_count} MFQ</span>}</form><div className="model-browser-list">{modelBrowser.data.map((directory) => <button disabled={busy} key={directory.id} onClick={() => void openModelDirectory(directory.id)} type="button"><Icon name="folder" /><span>{directory.name}</span>{directory.model_file_count > 0 && <b>{directory.model_file_count} MFQ</b>}</button>)}{modelBrowser.data.length === 0 && <p>{tr("这个文件夹中没有子文件夹。", "This folder has no subfolders.")}</p>}</div><footer><button onClick={() => setModelBrowserOpen(false)} type="button">{tr("取消", "Cancel")}</button><button className="primary" disabled={busy || !modelBrowser.current_id} onClick={() => void registerCurrentModelDirectory()} type="button">{tr("使用此文件夹", "Use this folder")}</button></footer></section></div>}

    </div>
  );
}
