/** 量化页面管理任务参数、校准产物、任务历史及日志订阅。 */
import { FormEvent, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useLocation } from 'react-router';
import type {
  ArtifactLineage,
  JobKindResource,
  JsonSchemaProperty,
  RuntimeLogEntry,
} from '../../api';
import { studioConfirm } from '../../studio';
import { schemaDefault, schemaType, isTerminalJob } from './jobSchema';
/** 按需加载量化资源，后台任务生命周期由共享运行时维持。 */
export function useQuantizationWorkspace() {
  const { tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const panelLabels = {
    collapse: tr('折叠面板', 'Collapse panel'),
    expand: tr('展开面板', 'Expand panel'),
  };
  const { jobs, addJob, refreshRuntime } = useRuntime();
  const location = useLocation();
  const [lineage, setLineage] = useState<ArtifactLineage[]>([]);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);
  const [selectedJobKind, setSelectedJobKind] = useState('');
  const [jobPayload, setJobPayload] = useState<Record<string, unknown>>({});
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [imatrixImporting, setImatrixImporting] = useState(false);
  const [pendingImatrix, setPendingImatrix] = useState('');
  const imatrixInputRef = useRef<HTMLInputElement | null>(null);
  const [jobLogs, setJobLogs] = useState<RuntimeLogEntry[]>([]);
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);
  const selectedJob = jobs.find((item) => item.id === selectedJobId) ?? null;
  const activeJobs = jobs.filter((item) => !isTerminalJob(item));
  const completedJobs = jobs.filter(isTerminalJob);
  const genericJobKinds = jobKinds;
  const selectedKind = jobKinds.find((item) => item.kind === selectedJobKind);
  const imatrixArtifacts = lineage.filter(
    (item) =>
      item.producer_kind === 'calibrate.imatrix' ||
      item.producer_kind === 'artifact.import' ||
      item.metadata?.media_type === 'application/x-mfq-imatrix' ||
      item.artifact_name.endsWith('.imatrix'),
  );
  useEffect(() => {
    const jobId = (location.state as { jobId?: string } | null)?.jobId;
    if (jobId) setSelectedJobId(jobId);
  }, [location.state]);
  useEffect(() => {
    let active = true;
    void api
      .jobKinds()
      .then((kinds) => {
        if (!active) return;
        setJobKinds(kinds);
        setSelectedJobKind((kind) => kind || kinds[0]?.kind || '');
      })
      .catch((cause) => {
        if (active) setError(errorMessage(cause));
      });
    return () => {
      active = false;
    };
  }, []);
  const completedVersion = completedJobs.map((job) => job.id + ':' + job.updated_at).join(',');
  useEffect(() => {
    let active = true;
    void api
      .artifactLineage()
      .then((items) => {
        if (active) setLineage(items);
      })
      .catch((cause) => {
        if (active) setError(errorMessage(cause));
      });
    return () => {
      active = false;
    };
  }, [completedVersion]);
  useEffect(() => {
    const properties =
      jobKinds.find((item) => item.kind === selectedJobKind)?.payload_schema.properties ?? {};
    setJobPayload(
      Object.fromEntries(
        Object.entries(properties).map(([name, property]) => [
          name,
          selectedJobKind === 'model.quantize' && name === 'imatrix' && pendingImatrix
            ? pendingImatrix
            : schemaDefault(property),
        ]),
      ),
    );
  }, [jobKinds, selectedJobKind, pendingImatrix]);
  useEffect(() => {
    setJobLogs([]);
    if (!selectedJobId) return;
    const controller = new AbortController();
    void api
      .jobEvents(selectedJobId)
      .then((entries) => {
        if (controller.signal.aborted) return;
        setJobLogs((current) =>
          [
            ...new Map([...entries, ...current].map((entry) => [entry.sequence, entry])).values(),
          ].sort((left, right) => left.sequence - right.sequence),
        );
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(errorMessage(cause));
      });
    void api
      .streamJobEvents(
        selectedJobId,
        (event) => {
          if (controller.signal.aborted || !event.message) return;
          setJobLogs((current) =>
            current.some((entry) => entry.sequence === event.sequence)
              ? current
              : [
                  ...current,
                  {
                    sequence: event.sequence,
                    level: event.level,
                    message: event.message || '',
                    fields: event.data,
                    created_at: event.created_at,
                  },
                ],
          );
        },
        controller.signal,
      )
      .catch((cause) => {
        if (!controller.signal.aborted) setError(errorMessage(cause));
      });
    return () => controller.abort();
  }, [selectedJobId]);
  /** 根据任务参数模式转换表单值。 */
  function updateJobPayload(name: string, property: JsonSchemaProperty, value: string | boolean) {
    const type = schemaType(property);
    let parsed: unknown = value;
    if (type === 'integer' || type === 'number') {
      parsed = value === '' ? null : Number(value);
    } else if (type === 'array') {
      parsed = String(value)
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => (property.items?.type === 'integer' ? Number(item) : item));
    }
    setJobPayload((current) => ({ ...current, [name]: parsed }));
  }
  /** 提交当前参数并选择新任务。 */
  async function submitJob(event: FormEvent) {
    event.preventDefault();
    if (!selectedJobKind) return;
    setBusy(true);
    try {
      const clean = Object.fromEntries(
        Object.entries(jobPayload).filter(([, value]) => value !== '' && value !== null),
      );
      const created = await api.createJob(selectedJobKind, clean);
      setSelectedJobId(created.id);
      addJob(created);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 上传并创建 Imatrix 导入任务。 */
  async function importImatrix(files: FileList | null) {
    const file = files?.[0];
    if (!file || imatrixImporting || busy) return;
    setImatrixImporting(true);
    try {
      const uploaded = await api.uploadMedia(file, 'application/x-mfq-imatrix');
      const destination = `artifacts/imatrix/${file.name.replace(/[^A-Za-z0-9_.-]+/g, '-')}`;
      const created = await api.createJob('artifact.import', {
        media_id: uploaded.media.id,
        destination,
        kind: 'imatrix',
      });
      setSelectedJobId(created.id);
      addJob(created);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImatrixImporting(false);
      if (imatrixInputRef.current) imatrixInputRef.current.value = '';
    }
  }
  /** 请求取消选中的后台任务。 */
  async function cancelSelectedJob() {
    if (!selectedJobId) return;
    try {
      const updated = await api.cancelJob(selectedJobId);
      addJob(updated);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }
  /** 为失败任务创建重试记录。 */
  async function retrySelectedJob() {
    if (!selectedJob || busy) return;
    setBusy(true);
    try {
      const retried = await api.retryJob(selectedJob.id);
      addJob(retried);
      setSelectedJobId(retried.id);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 移除终态任务记录并清理当前选择。 */
  async function deleteJobRecord(id: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.deleteJob(id);
      await refreshRuntime(false);
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
  /** 清理终态任务历史并同步共享运行时。 */
  async function clearCompletedJobRecords() {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await api.clearCompletedJobs();
      await refreshRuntime(false);
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
  /** 确认后删除选中任务的本地产物。 */
  async function removeSelectedArtifact() {
    const uri = String(selectedJob?.result?.artifact || '');
    if (!uri.startsWith('workspace://') || busy) return;
    if (!(await studioConfirm(tr('删除这个本地产物？', 'Delete this local artifact?')))) return;
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
  return {
    tr,
    busy,
    error,
    panelLabels,
    lineage,
    jobKinds,
    selectedJobKind,
    setSelectedJobKind,
    jobPayload,
    selectedJobId,
    setSelectedJobId,
    imatrixImporting,
    setPendingImatrix,
    imatrixInputRef,
    jobLogs,
    jobCleanupBusy,
    selectedJob,
    activeJobs,
    completedJobs,
    genericJobKinds,
    selectedKind,
    imatrixArtifacts,
    updateJobPayload,
    submitJob,
    importImatrix,
    cancelSelectedJob,
    retrySelectedJob,
    deleteJobRecord,
    clearCompletedJobRecords,
    removeSelectedArtifact,
  };
}
