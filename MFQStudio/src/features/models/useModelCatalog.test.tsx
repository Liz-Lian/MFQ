/** 验证模型目录的按需加载、策略传递及注册失败恢复。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api, type ModelArtifact } from '../../api';
import { useModelCatalog } from './useModelCatalog';

const state = vi.hoisted(() => ({
  ready: false,
  refreshRuntime: vi.fn(),
  setSelectedModel: vi.fn(),
}));
vi.mock('../../app/RuntimeProvider', () => ({
  useRuntime: () => ({
    ...state,
    runtime: null,
    models: [],
    instances: [],
    jobs: [],
    studio: null,
    selectedModel: '',
  }),
}));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en, contextSize: 8192 }),
}));
vi.mock('../../studio', () => ({ isStudio: () => false, selectLocalModelDirectory: vi.fn() }));

const artifact: ModelArtifact = {
  id: 'model-id',
  name: 'Local Model',
  architecture: 'test',
  format: 'mfq',
  shard_count: 1,
  total_bytes: 128,
  tensor_count: 1,
  record_count: 1,
  dtypes: [],
  complete: true,
  loadable: true,
  modified_at: '2026-09-23T00:00:00Z',
};

/** 提供真实路由上下文，避免把页面导航行为替换成无条件成功的桩。 */
function Wrapper({ children }: { children: ReactNode }) {
  return <MemoryRouter>{children}</MemoryRouter>;
}

beforeEach(() => {
  vi.restoreAllMocks();
  state.ready = false;
  state.refreshRuntime.mockReset().mockResolvedValue(undefined);
  state.setSelectedModel.mockReset();
  vi.spyOn(api, 'modelArtifacts').mockResolvedValue([artifact]);
});

describe('useModelCatalog', () => {
  it('waits for the platform before fetching catalog and filters by model name', async () => {
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    expect(api.modelArtifacts).not.toHaveBeenCalled();
    state.ready = true;
    rerender();
    await waitFor(() => expect(result.current.artifacts).toHaveLength(1));
    act(() => result.current.setModelFilter('missing'));
    expect(result.current.filteredArtifacts).toHaveLength(0);
    act(() => result.current.setModelFilter(' LOCAL '));
    expect(result.current.filteredArtifacts).toEqual([artifact]);
  });

  it('passes the page load policy and shared context size before selecting the loaded model', async () => {
    state.ready = true;
    const load = vi
      .spyOn(api, 'loadModel')
      .mockResolvedValue({ operation_id: 'load-1', status: 'accepted' });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await waitFor(() => expect(result.current.artifacts).toHaveLength(1));
    act(() => {
      result.current.setLoadPinned(true);
      result.current.setLoadIdleTtl(300);
    });
    await act(async () => result.current.loadArtifact(artifact.name));
    expect(load).toHaveBeenCalledWith('Local Model', 8192, 2048, {
      pin: true,
      idle_ttl_seconds: 300,
    });
    expect(state.refreshRuntime).toHaveBeenCalledWith(false);
    expect(state.setSelectedModel).toHaveBeenCalledWith('Local Model');
    expect(result.current.busy).toBe(false);
  });

  it('keeps the server directory dialog open after failed registration', async () => {
    state.ready = true;
    vi.spyOn(api, 'modelDirectories').mockResolvedValue({
      current_id: 'folder-1',
      current_name: 'models',
      current_path: '/models',
      parent_id: null,
      model_file_count: 1,
      data: [],
    });
    vi.spyOn(api, 'registerModelDirectory').mockRejectedValue(new Error('registration failed'));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    expect(result.current.modelBrowserOpen).toBe(true);
    await act(async () => result.current.registerCurrentModelDirectory());
    expect(result.current.error).toBe('registration failed');
    expect(result.current.modelBrowserOpen).toBe(true);
    expect(result.current.busy).toBe(false);
    expect(state.setSelectedModel).not.toHaveBeenCalled();
  });
});
