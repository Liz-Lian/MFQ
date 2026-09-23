/** 管理平台启动、共享推理实例与后台任务，不加载具体业务页面的数据。 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  api,
  setApiBaseUrl,
  setApiToken,
  type JobResource,
  type RuntimeStatus,
  type RuntimeModel,
  type RuntimeInstance,
  type RuntimeCapabilities,
  type RealtimeCapabilities,
  type VoiceOutputComponentStatus,
} from '../api';
import { studioStatus, studioCredential, startLocalStudio, type StudioStatus } from '../studio';
import { isRuntimeReady, runtimeSelectionNames } from '../features/runtime/modelSelection';
import { errorMessage } from './formatters';
import { useSettings } from '../features/settings/SettingsProvider';

interface RuntimeContextValue {
  runtime: RuntimeStatus | null;
  models: RuntimeModel[];
  instances: RuntimeInstance[];
  jobs: JobResource[];
  capabilities: RuntimeCapabilities | null;
  realtime: RealtimeCapabilities | null;
  voiceComponent: VoiceOutputComponentStatus | null;
  studio: StudioStatus | null;
  selectedModel: string;
  /** 更新全局选中模型，随后刷新该实例能力；聊天模块负责派生会话。 */
  setSelectedModel: (model: string) => void;
  /** 仅刷新跨页面共享的实例、模型、任务和能力，不请求评测或模型目录。 */
  refreshRuntime: (quiet?: boolean) => Promise<void>;
  /** 将新建任务并入共享列表，自动为运行中的任务建立事件订阅。 */
  addJob: (job: JobResource) => void;
  /** 重新读取平台地址与凭据，失效旧请求并通知业务重置连接缓存。 */
  reloadService: () => Promise<void>;
  connectionRevision: number;
  ready: boolean;
  loading: boolean;
  error: string | null;
}

const RuntimeContext = createContext<RuntimeContextValue | null>(null);

/** 为页面提供最小共享运行时，任务流跨路由存活，页面数据由各自模块维护。 */
export function RuntimeProvider({ children }: { children: ReactNode }) {
  const { setContextSize } = useSettings();
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [models, setModels] = useState<RuntimeModel[]>([]);
  const [instances, setInstances] = useState<RuntimeInstance[]>([]);
  const [jobs, setJobs] = useState<JobResource[]>([]);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [realtime, setRealtime] = useState<RealtimeCapabilities | null>(null);
  const [voiceComponent, setVoiceComponent] = useState<VoiceOutputComponentStatus | null>(null);
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionRevision, setConnectionRevision] = useState(0);
  const mounted = useRef(false);
  const requestVersion = useRef(0);
  const initializationVersion = useRef(0);
  const selectedRef = useRef(selectedModel);
  selectedRef.current = selectedModel;

  const refreshRuntime = useCallback(async (quiet = true) => {
    const version = ++requestVersion.current;
    if (!quiet) setLoading(true);
    try {
      const [nextInstances, nextJobs] = await Promise.all([api.runtimeInstances(), api.jobs(100)]);
      const instance = nextInstances.find(
        (item) => item.model === selectedRef.current && item.state !== 'failed',
      );
      const [statusResult, voiceResult] = await Promise.allSettled([
        api.runtimeStatus(instance?.id),
        api.voiceOutputComponent(),
      ]);
      if (statusResult.status !== 'fulfilled') throw statusResult.reason;
      const status = statusResult.value;
      const [modelResult, capabilityResult, realtimeResult] = await Promise.allSettled([
        isRuntimeReady(status.runtime_state)
          ? api.runtimeModels()
          : Promise.resolve<RuntimeModel[]>([]),
        isRuntimeReady(status.runtime_state)
          ? api.runtimeCapabilities(instance?.id)
          : Promise.resolve(null),
        isRuntimeReady(status.runtime_state) ? api.realtimeCapabilities() : Promise.resolve(null),
      ]);
      if (!mounted.current || version !== requestVersion.current) return;
      const nextModels = modelResult.status === 'fulfilled' ? modelResult.value : [];
      setInstances(nextInstances);
      setJobs(nextJobs);
      setRuntime(status);
      setModels(nextModels);
      setCapabilities(capabilityResult.status === 'fulfilled' ? capabilityResult.value : null);
      setRealtime(realtimeResult.status === 'fulfilled' ? realtimeResult.value : null);
      if (voiceResult.status === 'fulfilled') setVoiceComponent(voiceResult.value);
      const names = runtimeSelectionNames(nextModels, nextInstances, nextJobs);
      setSelectedModel((current) =>
        names.includes(current)
          ? current
          : typeof status.model === 'string' && names.includes(status.model)
            ? status.model
            : (names[0] ?? ''),
      );
      setError(null);
    } catch (cause) {
      if (mounted.current && version === requestVersion.current) setError(errorMessage(cause));
    } finally {
      if (mounted.current && version === requestVersion.current) setLoading(false);
    }
  }, []);

  const reloadService = useCallback(async () => {
    const version = ++initializationVersion.current;
    ++requestVersion.current;
    setReady(false);
    setLoading(true);
    try {
      let status = await studioStatus();
      if (!mounted.current || version !== initializationVersion.current) return;
      setStudio(status);
      if (status?.config.mode === 'local' && !status.reachable) {
        await startLocalStudio();
        status = await studioStatus();
      }
      let token = '';
      if (status) {
        try {
          token = await studioCredential();
        } catch (cause) {
          if (mounted.current && version === initializationVersion.current)
            setError(errorMessage(cause));
        }
      }
      if (!mounted.current || version !== initializationVersion.current) return;
      setApiBaseUrl(status?.service_url ?? '');
      setApiToken(token);
      setStudio(status);
      setSelectedModel('');
      selectedRef.current = '';
      setJobs([]);
      setInstances([]);
      setModels([]);
      setRuntime(null);
      setCapabilities(null);
      setConnectionRevision((current) => current + 1);
      await refreshRuntime();
      if (mounted.current && version === initializationVersion.current) setReady(true);
    } catch (cause) {
      if (mounted.current && version === initializationVersion.current) {
        setError(errorMessage(cause));
        setLoading(false);
      }
    }
  }, [refreshRuntime]);

  useEffect(() => {
    mounted.current = true;
    void reloadService();
    return () => {
      mounted.current = false;
      ++requestVersion.current;
      ++initializationVersion.current;
    };
  }, [reloadService]);

  useEffect(() => {
    if (ready && selectedModel) void refreshRuntime();
  }, [ready, selectedModel, refreshRuntime]);

  const activeJobIds = jobs
    .filter((job) => ['queued', 'running', 'cancelling'].includes(job.status))
    .map((job) => job.id)
    .sort()
    .join(',');
  useEffect(() => {
    const capacity = Number(runtime?.max_context);
    if (Number.isFinite(capacity) && capacity > 0) setContextSize(Math.floor(capacity));
  }, [runtime?.max_context, setContextSize]);
  useEffect(() => {
    if (!ready || !activeJobIds) return;
    const controller = new AbortController();
    for (const id of activeJobIds.split(',')) {
      void api
        .streamJobEvents(
          id,
          (event) => {
            if (controller.signal.aborted) return;
            const status =
              event.type === 'state' && typeof event.data.status === 'string'
                ? (event.data.status as JobResource['status'])
                : null;
            setJobs((current) =>
              current.map((job) =>
                job.id === id
                  ? {
                      ...job,
                      ...(status ? { status } : {}),
                      ...(typeof event.progress === 'number' ? { progress: event.progress } : {}),
                      updated_at: event.created_at,
                    }
                  : job,
              ),
            );
            if (status && ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(status))
              void refreshRuntime();
          },
          controller.signal,
        )
        .catch((cause) => {
          if (!controller.signal.aborted) setError(errorMessage(cause));
        });
    }
    return () => controller.abort();
  }, [ready, activeJobIds, connectionRevision, refreshRuntime]);

  const addJob = useCallback(
    (job: JobResource) =>
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]),
    [],
  );
  const value = useMemo(
    () => ({
      runtime,
      models,
      instances,
      jobs,
      capabilities,
      realtime,
      voiceComponent,
      studio,
      selectedModel,
      setSelectedModel,
      refreshRuntime,
      addJob,
      reloadService,
      connectionRevision,
      ready,
      loading,
      error,
    }),
    [
      runtime,
      models,
      instances,
      jobs,
      capabilities,
      realtime,
      voiceComponent,
      studio,
      selectedModel,
      refreshRuntime,
      addJob,
      reloadService,
      connectionRevision,
      ready,
      loading,
      error,
    ],
  );
  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

/** 读取平台与共享推理状态；必须位于 RuntimeProvider 内部。 */
export function useRuntime(): RuntimeContextValue {
  const value = useContext(RuntimeContext);
  if (!value) throw new Error('RuntimeProvider is missing');
  return value;
}
