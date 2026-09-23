/** 保留统一 API 导入入口，实际资源实现按领域拆分。 */
import { sessionsApi } from './shared/api/resources/sessions';
import { presetsApi } from './shared/api/resources/presets';
import { mediaApi } from './shared/api/resources/media';
import { modelsApi } from './shared/api/resources/models';
import { runtimeApi } from './shared/api/resources/runtime';
import { evaluationsApi } from './shared/api/resources/evaluations';
import { connectionsApi } from './shared/api/resources/connections';
import { jobsApi } from './shared/api/resources/jobs';

export * from './shared/api/types';
export {
  ApiError,
  setApiBaseUrl,
  setApiToken,
  getApiBaseUrl,
  runtimeRealtimeUrl,
} from './shared/api/client';
export { streamResponse } from './shared/api/responses';

/** 组合资源 API，兼容已有调用方并允许各业务逐步使用领域入口。 */
export const api = {
  ...sessionsApi,
  ...presetsApi,
  ...mediaApi,
  ...modelsApi,
  ...runtimeApi,
  ...evaluationsApi,
  ...connectionsApi,
  ...jobsApi,
};
