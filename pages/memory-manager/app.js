const bridge = window.AstrBotPluginPage;
const state = { data: {}, activeTab: "facts", busy: false, selectedGroupId: "" };

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

function setStatus(text = "", kind = "") {
  $("status").textContent = text;
  if (kind) $("status").dataset.kind = kind;
  else delete $("status").dataset.kind;
}

function setBusy(busy) {
  state.busy = busy;
  for (const button of document.querySelectorAll("button")) button.disabled = busy;
  $("scope").disabled = busy;
  $("summaryProvider").disabled = busy;
  $("summaryHours").disabled = busy;
}

function scopeLabel(scopeId) {
  const labels = state.data?.labels?.scopes || {};
  if (labels[scopeId]) return labels[scopeId];
  const value = String(scopeId || "");
  if (value.startsWith("botmesh:")) return value.slice(8) || "未命名群聊";
  return value ? "其他群聊" : "全部群聊";
}

function logicalGroupForScope(scopeId) {
  const value = String(scopeId || "");
  const mapped = state.data?.labels?.scope_groups?.[value];
  if (mapped) return String(mapped);
  return value.startsWith("botmesh:") ? value.slice(8) : "";
}

function groupLabelForScope(scopeId) {
  const logicalGroupId = logicalGroupForScope(scopeId);
  if (logicalGroupId) {
    return state.data?.labels?.groups?.[logicalGroupId] || logicalGroupId;
  }
  return scopeId ? "其他群聊" : "未分配群聊";
}

function logicalGroups() {
  const rows = Array.isArray(state.data.logical_groups) ? state.data.logical_groups : [];
  if (rows.length) return rows.map((item) => ({
    id: String(item.logical_group_id || ""),
    scopeId: String(item.scope_id || `botmesh:${item.logical_group_id || ""}`),
    label: String(item.display_name || item.logical_group_id || "未命名群聊"),
  })).filter((item) => item.id);
  return Object.entries(state.data?.labels?.groups || {}).map(([id, label]) => ({
    id,
    scopeId: `botmesh:${id}`,
    label: label || id,
  }));
}

function selectedGroup() {
  return logicalGroups().find((item) => item.id === state.selectedGroupId) || null;
}

function botLabel(botId) {
  const value = String(botId || "");
  const labels = state.data?.labels?.bots || {};
  return labels[value] || labels[`bot_${value}`] || labels[value.replace(/^bot_/, "")] || "未命名 Bot";
}

function statusLabel(value) {
  return ({ active: "已确认", inferred: "待确认", conflict: "有冲突", superseded: "已被纠正" })[value] || "记录";
}

function itemNode(kind, item) {
  const node = document.createElement("article");
  node.className = "item";
  const textKey = kind === "episode" ? "summary" : "text";
  let title;
  if (kind === "fact") title = `#${item.id} · ${statusLabel(item.status)} · 权威 ${item.authority}`;
  else if (kind === "private") {
    const account = botLabel(item.bot_id);
    const role = item.memory_key && item.memory_key !== item.bot_id ? item.memory_key : "";
    title = `#${item.id} · ${account}${role && role !== account ? ` · 记忆身份 ${role}` : ""} · ${item.kind || "记忆"}`;
  } else title = `#${item.id} · ${item.title || "未命名情景"}`;
  node.innerHTML = `<div class="item-head"><strong>${esc(title)}</strong><span class="meta">${esc(scopeLabel(item.scope_id))}</span></div>
    <textarea aria-label="${esc(title)}">${esc(item[textKey])}</textarea>
    <div class="actions"><button class="button button-secondary" data-save type="button">保存</button><button class="button button-quiet danger" data-delete type="button">删除</button></div>`;
  node.querySelector("[data-save]").onclick = async () => {
    setBusy(true);
    try {
      state.data = await bridge.apiPost("workspace/save", { operation: "update", kind, id: item.id, scope_id: selectedGroup()?.scopeId || String(item.scope_id || ""), values: { [textKey]: node.querySelector("textarea").value } });
      render();
      setStatus("已保存。", "success");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally { setBusy(false); }
  };
  node.querySelector("[data-delete]").onclick = async () => {
    const button = node.querySelector("[data-delete]");
    if (button.dataset.armed !== "1") {
      button.dataset.armed = "1";
      button.textContent = "再点一次确认删除";
      window.setTimeout(() => {
        if (button.dataset.armed === "1") {
          button.dataset.armed = "";
          button.textContent = "删除";
        }
      }, 4000);
      return;
    }
    button.dataset.armed = "";
    button.textContent = "删除";
    setBusy(true);
    try {
      state.data = await bridge.apiPost("workspace/save", { operation: "forget", kind, id: item.id, scope_id: selectedGroup()?.scopeId || String(item.scope_id || "") });
      render();
      setStatus("已删除。", "success");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally { setBusy(false); }
  };
  return node;
}

function groupedItems(items) {
  const groups = new Map();
  for (const item of items || []) {
    const groupId = logicalGroupForScope(item.scope_id) || `other:${item.scope_id || "unassigned"}`;
    if (!groups.has(groupId)) {
      groups.set(groupId, { label: groupLabelForScope(item.scope_id), items: [] });
    }
    groups.get(groupId).items.push(item);
  }
  return [...groups.values()].sort((left, right) => left.label.localeCompare(right.label, "zh-CN", { numeric: true }));
}

function renderList(rootId, items, kind) {
  const root = $(rootId);
  root.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = "<strong>这里还没有记录</strong><p>新对话会自动沉淀，也可以在上方手动总结。</p>";
    root.append(empty);
    return;
  }
  if (state.selectedGroupId) {
    items.forEach((item) => root.append(itemNode(kind, item)));
    return;
  }
  for (const group of groupedItems(items)) {
    const section = document.createElement("section");
    section.className = "memory-group";
    section.innerHTML = `<div class="memory-group-head"><h3>${esc(group.label)}</h3><span class="meta">${group.items.length} 条</span></div>`;
    const list = document.createElement("div");
    list.className = "memory-group-list";
    group.items.forEach((item) => list.append(itemNode(kind, item)));
    section.append(list);
    root.append(section);
  }
}

function renderCorrections(items) {
  const root = $("corrections");
  root.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = "<strong>暂无纠错</strong><p>用户明确更正的信息会显示在这里。</p>";
    root.append(empty);
    return;
  }
  const appendItem = (item, parent) => {
    const node = document.createElement("article");
    node.className = "item";
    node.innerHTML = `<div class="item-head"><strong>#${item.id} · 明确纠错</strong><span class="meta">${esc(scopeLabel(item.scope_id))}</span></div><p>旧：${esc(item.old_text || "未指定")}</p><p>新：${esc(item.new_text)}</p><p class="meta">${esc(item.reason || "用户明确纠正")}</p>`;
    parent.append(node);
  };
  if (state.selectedGroupId) {
    items.forEach((item) => appendItem(item, root));
    return;
  }
  for (const group of groupedItems(items)) {
    const section = document.createElement("section");
    section.className = "memory-group";
    section.innerHTML = `<div class="memory-group-head"><h3>${esc(group.label)}</h3><span class="meta">${group.items.length} 条</span></div>`;
    const list = document.createElement("div");
    list.className = "memory-group-list";
    group.items.forEach((item) => appendItem(item, list));
    section.append(list);
    root.append(section);
  }
}

function renderSchedules(items) {
  const root = $("schedules");
  root.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = "<strong>暂无日程</strong><p>Dynamic Life 成功生成并写入 Memory 后会显示在这里。</p>";
    root.append(empty);
    return;
  }
  const appendItem = (item, parent) => {
    const node = document.createElement("article");
    node.className = "item schedule-item";
    const account = botLabel(item.bot_id);
    const role = String(item.memory_key || account || "未命名角色");
    const businessDate = String(item.business_date || "未知日期");
    const createdAt = Number(item.created_at || 0);
    const createdLabel = createdAt
      ? new Date(createdAt * 1000).toLocaleString("zh-CN", { hour12: false })
      : "时间未知";
    node.innerHTML = `<div class="item-head"><strong>#${Number(item.id || 0)} · ${esc(businessDate)} · ${esc(role)}</strong><span class="meta">${esc(scopeLabel(item.scope_id))}</span></div>
      <div class="schedule-meta">记忆身份：${esc(role)} · 承载账号：${esc(account)} · 写入：${esc(createdLabel)}</div>`;
    const content = document.createElement("pre");
    content.className = "schedule-content";
    content.textContent = String(item.schedule_text || item.assistant_message || "");
    node.append(content);
    parent.append(node);
  };
  if (state.selectedGroupId) {
    items.forEach((item) => appendItem(item, root));
    return;
  }
  for (const group of groupedItems(items)) {
    const section = document.createElement("section");
    section.className = "memory-group";
    section.innerHTML = `<div class="memory-group-head"><h3>${esc(group.label)}</h3><span class="meta">${group.items.length} 条</span></div>`;
    const list = document.createElement("div");
    list.className = "memory-group-list";
    group.items.forEach((item) => appendItem(item, list));
    section.append(list);
    root.append(section);
  }
}

function renderScope() {
  const groups = logicalGroups().sort((left, right) => left.label.localeCompare(right.label, "zh-CN", { numeric: true }));
  if (state.selectedGroupId && !groups.some((item) => item.id === state.selectedGroupId)) state.selectedGroupId = "";
  const options = [
    { value: "", label: "全部群聊" },
    ...groups.map((item) => ({ value: item.id, label: item.label })),
  ];
  $("scope").replaceChildren(...options.map((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    return option;
  }));
  $("scope").value = state.selectedGroupId;
}

function renderBots() {
  const control = $("importBot");
  const current = control.value;
  const bots = Array.isArray(state.data.logical_bots) ? state.data.logical_bots : [];
  control.replaceChildren(...bots.map((item) => {
    const option = document.createElement("option");
    option.value = item.bot_id;
    option.textContent = item.display_name || botLabel(item.bot_id);
    return option;
  }));
  if (!bots.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "未读取到 Bot 名称";
    control.append(option);
  }
  if ([...control.options].some((item) => item.value === current)) control.value = current;
}

function renderProviders() {
  const control = $("summaryProvider");
  const current = control.value || state.data.configured_provider_id || "";
  const providers = Array.isArray(state.data.providers) ? state.data.providers : [];
  control.replaceChildren(...providers.map((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name || "已配置模型";
    return option;
  }));
  if (!providers.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "跟随当前群聊模型";
    control.append(option);
  }
  if ([...control.options].some((item) => item.value === current)) control.value = current;
}

function render() {
  const scroller = document.scrollingElement || document.documentElement;
  const scrollTop = scroller ? scroller.scrollTop : 0;
  renderScope();
  renderProviders();
  renderBots();
  const groupId = state.selectedGroupId;
  const pick = (rows) => (rows || []).filter((item) => !groupId || logicalGroupForScope(item.scope_id) === groupId);
  renderList("facts", pick(state.data.facts), "fact");
  renderList("private", pick(state.data.private_memories), "private");
  renderList("episodes", pick(state.data.episodes), "episode");
  renderSchedules(pick(state.data.schedules));
  renderCorrections(pick(state.data.corrections));
  if (scroller) {
    requestAnimationFrame(() => scroller.scrollTo(0, scrollTop));
  }
}

function switchTab(name) {
  state.activeTab = name;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  for (const panelName of ["facts", "private", "episodes", "schedules", "corrections"]) {
    $(`${panelName}Panel`).hidden = panelName !== name;
  }
}

async function load() {
  setBusy(true);
  setStatus("正在读取…");
  try {
    state.data = await bridge.apiGet("workspace");
    render();
    setStatus("记忆已同步。", "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally { setBusy(false); }
}

document.querySelectorAll(".tab").forEach((tab) => tab.onclick = () => switchTab(tab.dataset.tab));
$("scope").onchange = () => {
  state.selectedGroupId = $("scope").value;
  render();
};
$("reload").onclick = load;
$("addFact").onclick = async () => {
  const text = $("newFact").value.trim();
  const groups = selectedGroup() ? [selectedGroup()] : logicalGroups();
  if (!groups.length || !text) { setStatus("没有可用逻辑群，或事实内容为空。", "error"); return; }
  setBusy(true);
  try {
    for (const group of groups) {
      state.data = await bridge.apiPost("workspace/save", { operation: "add_fact", scope_id: group.scopeId, text });
    }
    $("newFact").value = "";
    render();
    setStatus(`事实已固定到 ${groups.length} 个逻辑群。`, "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally { setBusy(false); }
};

$("summarize").onclick = async () => {
  const selected = selectedGroup();
  const groups = selected ? [selected] : logicalGroups();
  if (!groups.length) { setStatus("没有读取到 BotMesh 逻辑群，请检查 BotMesh 群映射。", "error"); return; }
  setBusy(true);
  const completed = [];
  const failed = [];
  try {
    for (let index = 0; index < groups.length; index += 1) {
      const group = groups[index];
      setStatus(`正在总结 ${index + 1}/${groups.length}：“${group.label}”…`);
      try {
        state.data = await bridge.apiPost("workspace/summarize", {
          logical_group_id: group.id,
          scope_id: group.scopeId,
          provider_id: $("summaryProvider").value,
          hours: Number($("summaryHours").value || 24),
          limit: 160,
        });
        completed.push({ group, count: state.data.source_message_count || 0 });
      } catch (error) {
        failed.push(`${group.label}：${error.message || String(error)}`);
      }
    }
    if (!completed.length) throw new Error(failed.join("；") || "没有群聊可总结");
    render();
    switchTab("episodes");
    const total = completed.reduce((sum, item) => sum + item.count, 0);
    setStatus(`已完成 ${completed.length}/${groups.length} 个群，共总结 ${total} 条消息${failed.length ? `；跳过：${failed.join("；")}` : ""}。`, "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally { setBusy(false); }
};

$("importKind").onchange = () => {
  $("importBot").disabled = state.busy || $("importKind").value !== "private";
};

$("importMemory").onclick = async () => {
  const text = $("importText").value.trim();
  const selected = selectedGroup();
  const groups = selected ? [selected] : logicalGroups();
  const importKind = $("importKind").value;
  const botId = $("importBot").value;
  if (!groups.length || !text) { setStatus("没有可用逻辑群，或导入内容为空。", "error"); return; }
  if (importKind === "private" && !botId) { setStatus("请选择记忆所属 Bot。", "error"); return; }
  setBusy(true);
  let imported = 0;
  try {
    for (const group of groups) {
      setStatus(`正在向“${group.label}”导入过去记忆…`);
      state.data = await bridge.apiPost("workspace/import", {
        logical_group_id: group.id,
        scope_id: group.scopeId,
        import_kind: importKind,
        bot_id: botId,
        text,
      });
      imported += Number(state.data.imported || 0);
    }
    $("importText").value = "";
    render();
    if (importKind === "private") switchTab("private");
    else if (importKind === "episode") switchTab("episodes");
    else switchTab("facts");
    setStatus(`已向 ${groups.length} 个逻辑群导入 ${imported} 条记忆。`, "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally { setBusy(false); $("importKind").onchange(); }
};

$("migrateLegacy").onclick = async () => {
  if (!window.confirm("整理会把旧平台会话键和平台账号 ID 转成 BotMesh 逻辑群与 Bot 节点。执行前会自动备份数据库，是否继续？")) return;
  setBusy(true);
  setStatus("正在整理现有旧记忆…");
  try {
    state.data = await bridge.apiPost("workspace/migrate", {});
    render();
    const migration = state.data.migration || {};
    setStatus(`整理完成：迁移 ${migration.moved_scope_rows || 0} 条作用域、${migration.moved_identity_rows || 0} 条 Bot 身份。`, "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally { setBusy(false); $("importKind").onchange(); }
};

if (!bridge) {
  setStatus("未检测到 AstrBot Plugin Page 环境，请从插件详情页打开。", "error");
  setBusy(true);
} else {
  (async () => { await bridge.ready(); await load(); })();
}
