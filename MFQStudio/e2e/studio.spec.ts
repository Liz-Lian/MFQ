/** 验证真实浏览器中的发送、恢复、取消、输入法与弹窗交互，并检查响应式布局。 */
import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('发送流式回答并完成历史同步', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/#/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('Explain this project');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.locator('.message-assistant strong')).toHaveText('formatted content');
  await expect(input).toBeEnabled();
  await expect(page.locator('.message-assistant')).toHaveCount(1);
  await expect.poll(() => state.submissions).toBe(1);
  expect(state.unexpected).toEqual([]);
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: testInfo.outputPath('chat.png'), fullPage: true });
});

test('输入法确认不会误发，Shift Enter 保留换行', async ({ page }) => {
  const state = await mockStudioServer(page);
  await page.goto('/#/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(input).toBeEnabled();
  await input.fill('中文候选');
  await input.dispatchEvent('compositionstart');
  await input.dispatchEvent('keydown', {
    key: 'Enter',
    code: 'Enter',
    isComposing: true,
    keyCode: 229,
  });
  await input.dispatchEvent('compositionend');
  await expect(input).toHaveValue('中文候选');
  expect(state.submissions).toBe(0);
  await input.press('Shift+Enter');
  await expect(input).toHaveValue('中文候选\n');
  expect(state.submissions).toBe(0);
});

test('历史同步失败保留回答且恢复时不重复生成', async ({ page }) => {
  const state = await mockStudioServer(page, { failFirstSync: true });
  await page.goto('/#/chat');
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Keep this response');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  const retry = page.getByRole('button', { name: 'Synchronize response', exact: true });
  await expect(retry).toBeVisible();
  await expect(page.getByText('formatted content', { exact: true })).toBeVisible();
  await retry.click();
  await expect(retry).toBeHidden();
  await expect(page.locator('.message-assistant')).toHaveCount(1);
  expect(state.submissions).toBe(1);
});

test('停止挂起生成后恢复输入', async ({ page }) => {
  const state = await mockStudioServer(page, { waitForCancel: true });
  await page.goto('/#/chat');
  const input = page.getByRole('textbox', { name: 'Message', exact: true });
  await input.fill('Stop this request');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect.poll(() => state.submissions).toBe(1);
  await page.getByRole('button', { name: 'Stop generation', exact: true }).click();
  await expect(input).toBeEnabled();
  expect(state.cancellations).toBe(1);
  expect(state.submissions).toBe(1);
});

test('路由导航及模型目录弹窗键盘焦点', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  await page.goto('/#/models');
  const addModel = page.getByRole('button', { name: 'Add model', exact: true }).first();
  await expect(addModel).toBeEnabled();
  await addModel.click();
  const dialog = page.getByRole('dialog', { name: 'Choose model folder' });
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Tab');
  expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-dialog.png'), fullPage: true });
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(addModel).toBeFocused();
  if (testInfo.project.name === 'mobile')
    await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Overview', exact: true }).click();
  await expect(page).toHaveURL('/#/');
  await expect(page.getByRole('main')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(state.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('overview.png'), fullPage: true });
});
