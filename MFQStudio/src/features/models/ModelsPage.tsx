/** 模型页负责资产加载、实例策略和模型目录浏览。 */
import { useSettings } from '../settings/SettingsProvider';
import {
  Icon,
  ScreenHeader,
  SectionLabel,
  TMPanel,
  SettingRow,
  EmptyPanel,
} from '../../app/display';
import { formatNumber } from '../../app/formatters';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';

/** 仅在模型页请求资产目录，页面独立持有加载策略和目录选择状态。 */
export function ModelsPage() {
  const catalog = useModelCatalog();
  const { tr } = useSettings();
  const {
    runtime,
    model,
    artifacts,
    busy,
    error,
    ready,
    instances,
    availableModelNames,
    modelFilter,
    setModelFilter,
    filteredInstances,
    filteredArtifacts,
    loadPinned,
    setLoadPinned,
    loadIdleTtl,
    setLoadIdleTtl,
    selectModel,
    openStudioPage,
    chooseModelDirectory,
    unloadInstance,
    loadArtifact,
  } = catalog;
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('模型', 'Models')}
        subtitle={tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.')}
        trailing={
          <button
            disabled={busy || !ready}
            onClick={() => void chooseModelDirectory()}
            type="button"
          >
            <Icon name="plus" size={14} />
            {tr('添加模型', 'Add model')}
          </button>
        }
      />
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <div className="model-workbench-summary">
        <div>
          <span>{tr('运行中的模型', 'Loaded models')}</span>
          <strong>{availableModelNames.length}</strong>
          <small>{tr('可直接用于对话', 'Ready for chat')}</small>
        </div>
        <div>
          <span>{tr('本地检查点', 'Local checkpoints')}</span>
          <strong>{artifacts.length}</strong>
          <small>{tr('已登记到 MFQ', 'Registered in MFQ')}</small>
        </div>
        <div>
          <span>{tr('当前对话模型', 'Chat model')}</span>
          <strong title={model || undefined}>{model || tr('未选择', 'None')}</strong>
          <small>
            {model
              ? tr('切换会话模型不会重新注册资产', 'Switching keeps the registered asset')
              : tr('加载后从这里选择', 'Choose one after loading')}
          </small>
        </div>
        <div className="model-workbench-links">
          <button onClick={() => openStudioPage('lab', 'models')} type="button">
            <Icon name="download" size={13} />
            {tr('打开模型仓库', 'Open model hub')}
          </button>
          <button onClick={() => openStudioPage('lab', 'quantization')} type="button">
            <Icon name="memory" size={13} />
            {tr('去量化', 'Quantize')}
          </button>
        </div>
      </div>
      <div className="model-catalog-toolbar">
        <div>
          <strong>{tr('模型资产', 'Model assets')}</strong>
          <span>{tr('注册、加载和切换对话模型', 'Register, load, and switch chat models')}</span>
        </div>
        <label>
          <span aria-hidden="true">/</span>
          <input
            aria-label={tr('筛选模型', 'Filter models')}
            onChange={(event) => setModelFilter(event.target.value)}
            placeholder={tr('按名称筛选', 'Filter by name')}
            value={modelFilter}
          />
        </label>
      </div>
      <SectionLabel
        title={tr('已加载模型', 'Loaded models')}
        subtitle={tr(
          `${availableModelNames.length} 个可用于推理`,
          `${availableModelNames.length} available for inference`,
        )}
      />
      {filteredInstances.length > 0 ? (
        <TMPanel className="model-catalog-panel loaded-model-panel">
          <div className="model-list">
            {filteredInstances.map((instance) => {
              const ready = instance.state === 'ready' || instance.state === 'busy';
              const selected = instance.model === model;
              const stateLabel =
                instance.state === 'loading'
                  ? tr('加载中', 'Loading')
                  : instance.state === 'unloading'
                    ? tr('卸载中', 'Unloading')
                    : instance.state === 'failed'
                      ? tr('失败', 'Failed')
                      : instance.state === 'busy'
                        ? tr('使用中', 'Busy')
                        : tr('就绪', 'Ready');
              return (
                <div className="model-row" key={instance.id}>
                  <span
                    className={
                      instance.state === 'failed'
                        ? 'model-state failed'
                        : ready
                          ? 'model-state active'
                          : 'model-state'
                    }
                  />
                  <div>
                    <strong>{instance.model}</strong>
                    <small>
                      {stateLabel} · {formatNumber(instance.context_size)} ctx
                      {instance.pinned
                        ? ` · ${tr('固定', 'Pinned')}`
                        : instance.idle_ttl_seconds != null
                          ? ` · TTL ${instance.idle_ttl_seconds}s`
                          : ''}
                    </small>
                  </div>
                  <div className="model-row-actions">
                    {ready && (
                      <button
                        className={selected ? 'selected' : ''}
                        disabled={busy || selected}
                        onClick={() => selectModel(instance.model)}
                        type="button"
                      >
                        {selected ? tr('当前', 'Current') : tr('用于对话', 'Use in chat')}
                      </button>
                    )}
                    <button
                      disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)}
                      type="button"
                    >
                      {tr('卸载', 'Unload')}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="memory"
          title={tr(
            modelFilter ? '没有匹配的已加载模型' : '当前没有已加载模型',
            modelFilter ? 'No loaded model matches' : 'No models loaded',
          )}
          message={tr(
            '从本地检查点加载一个模型后即可开始对话。',
            'Load a local checkpoint to start chatting.',
          )}
          action={
            <button
              className="primary"
              disabled={busy}
              onClick={() => void chooseModelDirectory()}
              type="button"
            >
              <Icon name="folder" size={14} />
              {tr('添加模型', 'Add model')}
            </button>
          }
        />
      )}
      <TMPanel className="model-catalog-panel">
        <div className="panel-heading">
          <div>
            <h2>{tr('加载策略', 'Load policy')}</h2>
            <p>
              {tr('控制模型的驻留与自动卸载。', 'Control model residency and automatic unloading.')}
            </p>
          </div>
        </div>
        <div className="setting-list model-policy-panel">
          <SettingRow
            title={tr('固定到内存', 'Pin in memory')}
            detail={tr('跳过 LRU 与空闲卸载', 'Skip LRU and idle eviction')}
            trailing={
              <input
                aria-label={tr('固定到内存', 'Pin in memory')}
                checked={loadPinned}
                onChange={(event) => setLoadPinned(event.target.checked)}
                type="checkbox"
              />
            }
          />
          <SettingRow
            title={tr('空闲卸载', 'Idle unload')}
            detail={
              loadPinned
                ? tr('固定模型不使用 TTL', 'Ignored while pinned')
                : tr('每次使用后重新计时', 'Resets after each use')
            }
            trailing={
              <select
                disabled={loadPinned}
                onChange={(event) =>
                  setLoadIdleTtl(event.target.value ? Number(event.target.value) : null)
                }
                value={loadIdleTtl ?? ''}
              >
                <option value="">{tr('永不', 'Never')}</option>
                <option value="300">5 min</option>
                <option value="900">15 min</option>
                <option value="3600">1 h</option>
              </select>
            }
          />
        </div>
      </TMPanel>
      <SectionLabel
        title={tr('本地检查点', 'Local checkpoints')}
        subtitle={`${artifacts.length} ${tr('个本地模型', 'local models')}`}
      />
      {filteredArtifacts.length > 0 ? (
        <TMPanel className="model-catalog-panel model-library-panel">
          <div className="model-list">
            {filteredArtifacts.map((item) => {
              const instance = instances.find(
                (candidate) => candidate.model === item.name && candidate.state !== 'failed',
              );
              const loaded = Boolean(instance) || item.name === runtime?.model;
              const policy = instance?.pinned
                ? tr('固定', 'Pinned')
                : instance?.idle_ttl_seconds != null
                  ? `TTL ${instance.idle_ttl_seconds}s`
                  : null;
              return (
                <div className="model-row" key={item.id}>
                  <span
                    className={
                      loaded
                        ? 'model-state active'
                        : item.loadable
                          ? 'model-state'
                          : 'model-state failed'
                    }
                  />
                  <div>
                    <strong>{item.name}</strong>
                    <small>
                      {item.architecture} · {item.shard_count} shards ·{' '}
                      {formatNumber(item.total_bytes / 2 ** 30, 1)} GB{policy ? ` · ${policy}` : ''}
                    </small>
                  </div>
                  {instance ? (
                    <button
                      disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)}
                      type="button"
                    >
                      {tr('卸载', 'Unload')}
                    </button>
                  ) : loaded ? (
                    <em>{tr('已加载', 'Loaded')}</em>
                  ) : !item.loadable ? (
                    <em className="failed" title={item.error || undefined}>
                      {item.complete && item.format === 'hf'
                        ? tr('需先转换', 'Convert first')
                        : tr('不可用', 'Invalid')}
                    </em>
                  ) : (
                    <button
                      disabled={busy}
                      onClick={() => void loadArtifact(item.name)}
                      type="button"
                    >
                      {tr('加载', 'Load')}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="folder"
          title={tr(
            modelFilter ? '没有匹配的本地模型' : '还没有本地模型',
            modelFilter ? 'No local model matches' : 'No local models yet',
          )}
          message={tr('添加一个模型文件夹即可开始。', 'Add a model folder to get started.')}
          action={
            <button
              className="primary"
              disabled={busy}
              onClick={() => void chooseModelDirectory()}
              type="button"
            >
              <Icon name="folder" size={14} />
              {tr('选择模型文件夹', 'Choose model folder')}
            </button>
          }
        />
      )}
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
