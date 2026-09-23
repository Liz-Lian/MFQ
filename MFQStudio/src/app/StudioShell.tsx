/** 应用外壳只负责导航、共享状态摘要和路由错误边界，不拥有业务表单与请求。 */
import { Component, Suspense, useEffect, type ReactNode } from 'react';
import { useLocation, useNavigate, Outlet } from 'react-router';
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import { useUiStore } from '../stores/uiStore';
import { ToastContainer } from '../shared/ui/Toast';
import { Icon } from './display';
import { FailurePage } from './FailurePage';
import { LoadingPage } from './LoadingPage';
import { formatNumber } from './formatters';
import { runtimeModelNames } from '../features/runtime/modelSelection';
import {
  resolveStudioLocation,
  isStudioPath,
  dashboardPath,
  labPath,
  type DashboardPage,
  type LabPage,
} from '../navigation';

/** 隔离单个业务路由的渲染错误，切换路径后恢复其他页面。 */
class PageErrorBoundary extends Component<{ children: ReactNode }, { detail: string | null }> {
  state = { detail: null as string | null };
  static getDerivedStateFromError(error: unknown) {
    return { detail: error instanceof Error ? error.message : String(error) };
  }
  render() {
    if (this.state.detail !== null)
      return (
        <FailurePage
          detail={this.state.detail}
          kind="render"
          onRetry={() => this.setState({ detail: null })}
        />
      );
    return this.props.children;
  }
}

/** 渲染固定导航和嵌套路由出口，页面不通过外壳传递业务状态。 */
export function StudioShell() {
  const location = useLocation();
  const navigate = useNavigate();
  const { tr } = useSettings();
  const {
    runtime,
    selectedModel: model,
    models,
    instances,
    loading: selectedModelLoading,
    error,
    ready,
    reloadService,
  } = useRuntime();
  const currentLocation = resolveStudioLocation(location.pathname);
  const { dashboardPage, labPage } = currentLocation;
  const view = isStudioPath(location.pathname) ? currentLocation.view : 'not-found';
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const closeSidebar = useUiStore((state) => state.closeSidebar);
  const openSidebar = useUiStore((state) => state.openSidebar);
  const availableModelNames = runtimeModelNames(models, instances);
  const selectedModelAvailable = availableModelNames.includes(model);
  const pageAvailable = ready || view === 'not-found' || location.pathname === '/runtime' || location.pathname === '/settings';
  useEffect(() => {
    closeSidebar();
    const title =
      location.pathname === '/' ? 'Overview' : location.pathname.slice(1).replaceAll('-', ' ');
    document.title = `${title} · MFQ Studio`;
  }, [location.pathname, closeSidebar]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === ',') {
        event.preventDefault();
        navigate('/settings');
      }
      if (event.key === 'Escape') closeSidebar();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [navigate, closeSidebar]);
  /** 从导航项跳转到业务路由并关闭移动侧栏。 */
  function openStudioPage(nextView: 'dashboard' | 'lab', page: DashboardPage | LabPage) {
    navigate(
      nextView === 'dashboard' ? dashboardPath(page as DashboardPage) : labPath(page as LabPage),
    );
    closeSidebar();
  }
  /** 打开聊天页，生成生命周期由 ChatProvider 保留。 */
  function openChatPage() {
    navigate('/chat');
    closeSidebar();
  }
  /** 打开服务器连接设置。 */
  function openServerPage() {
    navigate('/runtime');
    closeSidebar();
  }
  /** 打开偏好与生成设置。 */
  function openSettings() {
    navigate('/settings');
    closeSidebar();
  }
  return (
    <div className="app-shell">
      <ToastContainer />
      <a className="skip-link" href="#studio-main">
        {tr('跳到主要内容', 'Skip to main content')}
      </a>
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} id="studio-sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div>
            <strong>MFQ</strong>
            <span>Studio</span>
          </div>
        </div>
        <div className="sidebar-scroll">
          <nav className="sectioned-nav" aria-label={tr('推理', 'Inference')}>
            <section>
              <div className="sidebar-group-label">{tr('推理', 'Inference')}</div>
              <button
                aria-current={
                  view === 'dashboard' && dashboardPage === 'overview' ? 'page' : undefined
                }
                className={view === 'dashboard' && dashboardPage === 'overview' ? 'active' : ''}
                onClick={() => openStudioPage('dashboard', 'overview')}
                type="button"
              >
                <Icon name="gauge" />
                {tr('概览', 'Overview')}
                {Number(runtime?.active_requests || 0) > 0 && (
                  <span>{formatNumber(runtime?.active_requests || 0)}</span>
                )}
              </button>
              <button
                aria-current={
                  view === 'dashboard' && dashboardPage === 'models' ? 'page' : undefined
                }
                className={view === 'dashboard' && dashboardPage === 'models' ? 'active' : ''}
                onClick={() => openStudioPage('dashboard', 'models')}
                type="button"
              >
                <Icon name="folder" />
                {tr('模型', 'Models')}
              </button>
              <button
                aria-current={
                  view === 'dashboard' && dashboardPage === 'connections' ? 'page' : undefined
                }
                className={view === 'dashboard' && dashboardPage === 'connections' ? 'active' : ''}
                onClick={openServerPage}
                type="button"
              >
                <Icon name="server-rack" />
                {tr('服务器', 'Server')}
              </button>
              <button
                aria-current={
                  view === 'dashboard' && dashboardPage === 'cache' ? 'page' : undefined
                }
                className={view === 'dashboard' && dashboardPage === 'cache' ? 'active' : ''}
                onClick={() => openStudioPage('dashboard', 'cache')}
                type="button"
              >
                <Icon name="memory" />
                {tr('资源', 'Resources')}
              </button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr('交互', 'Playground')}</div>
              <button
                aria-current={view === 'chat' ? 'page' : undefined}
                className={view === 'chat' ? 'active' : ''}
                onClick={openChatPage}
                type="button"
              >
                <Icon name="chat" />
                {tr('对话', 'Chat')}
              </button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr('模型工具', 'Model tools')}</div>
              <button
                className={view === 'lab' && labPage === 'models' ? 'active' : ''}
                onClick={() => openStudioPage('lab', 'models')}
                type="button"
              >
                <Icon name="download" />
                {tr('模型仓库', 'Model hub')}
              </button>
              <button
                className={view === 'lab' && labPage === 'evaluations' ? 'active' : ''}
                onClick={() => openStudioPage('lab', 'evaluations')}
                type="button"
              >
                <Icon name="activity" />
                {tr('评测与数据集', 'Evaluations')}
              </button>
              <button
                className={view === 'lab' && labPage === 'quantization' ? 'active' : ''}
                onClick={() => openStudioPage('lab', 'quantization')}
                type="button"
              >
                <Icon name="memory" />
                {tr('量化工作台', 'Quantization')}
              </button>
            </section>
            <section>
              <div className="sidebar-group-label">{tr('系统', 'System')}</div>
              <button
                aria-current={view === 'dashboard' && dashboardPage === 'logs' ? 'page' : undefined}
                className={view === 'dashboard' && dashboardPage === 'logs' ? 'active' : ''}
                onClick={() => openStudioPage('dashboard', 'logs')}
                type="button"
              >
                <Icon name="activity" />
                {tr('日志', 'Logs')}
              </button>
              <button
                aria-current={
                  view === 'dashboard' && dashboardPage === 'settings' ? 'page' : undefined
                }
                className={view === 'dashboard' && dashboardPage === 'settings' ? 'active' : ''}
                onClick={openSettings}
                type="button"
              >
                <Icon name="settings" />
                {tr('设置', 'Settings')}
              </button>
            </section>
          </nav>
        </div>
        <button
          className="sidebar-runtime-card"
          onClick={() => openStudioPage('dashboard', 'overview')}
          type="button"
        >
          <span
            className={`runtime-dot ${Number(runtime?.active_requests || 0) > 0 ? 'busy' : selectedModelAvailable ? 'ready' : selectedModelLoading ? 'busy' : 'idle'}`}
          />
          <span>
            <strong>{model || tr('服务空闲', 'Server idle')}</strong>
            <small>
              {availableModelNames.length > 1
                ? tr(
                    `${availableModelNames.length} 个模型已加载`,
                    `${availableModelNames.length} models loaded`,
                  )
                : selectedModelAvailable
                  ? `${formatNumber(runtime?.active_requests || 0)} ${tr('个活动请求', 'active requests')}`
                  : selectedModelLoading
                    ? tr('模型加载中', 'Model loading')
                    : tr('选择模型以开始', 'Choose a model to begin')}
            </small>
          </span>
          <Icon name="activity" size={14} />
        </button>
      </aside>
      <button
        aria-controls="studio-sidebar"
        aria-expanded={sidebarOpen}
        aria-label={tr('打开侧栏', 'Open sidebar')}
        className="mobile-menu-trigger"
        onClick={openSidebar}
        type="button"
      >
        <Icon name="menu" size={17} />
      </button>
      <button
        aria-label={tr('关闭侧栏', 'Close sidebar')}
        className={`mobile-scrim ${sidebarOpen ? 'open' : ''}`}
        onClick={closeSidebar}
        type="button"
      />

      <main
        className={view === 'chat' ? 'workspace chat-workspace' : 'workspace'}
        id="studio-main"
        tabIndex={-1}
      >
        {error && (
          <div role="alert" className="error-banner">
            {error}
          </div>
        )}
        {pageAvailable ? (
          <PageErrorBoundary key={location.pathname}>
            <Suspense fallback={<LoadingPage />}>
              <Outlet />
            </Suspense>
          </PageErrorBoundary>
        ) : (
          error ? (
            <FailurePage detail={error} kind="connection" onRetry={() => void reloadService()} />
          ) : <LoadingPage />
        )}
      </main>
    </div>
  );
}
