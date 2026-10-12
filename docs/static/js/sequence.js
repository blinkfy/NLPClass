import { api, node, notice } from "./api.js";
import { clearAlgorithm, renderTrace, renderModelInfo, resizeAlgorithm, exportProcess } from "./algorithm-view.js";
import {
  loadParameters,
  loadMetrics,
  loadComparison,
  resetCharts,
  resizeCharts,
  exportChart,
} from "./charts.js";

const $ = (id) => document.getElementById(id);
const categories = [
  "OOV",
  "Proper noun",
  "Ambiguity",
  "Number/date",
  "English mixed",
  "Other",
];
let activeTab = "interactive",
  errorPage = 1,
  generation = 0;
const currentRun = () => $("run-select").value;
const query = () => `run_id=${encodeURIComponent(currentRun())}`;
const extensionRuns = new Map();
const segmentationRun = () => extensionRuns.get($("segmentation-model").value)?.run_id || currentRun();

function updateActiveRun() {
  const option = $("run-select").selectedOptions[0];
  const extension = extensionRuns.get($("segmentation-model").value);
  $("active-run").textContent = extension
    ? `分词 · ${extension.model} · MSR Run · ${extension.run_id} ｜ NER/原有评测 Run · ${currentRun()}`
    : currentRun() ? `模型 · ${option.dataset.model} · Run · ${currentRun()}` : "";
}

function showTab(key) {
  activeTab = key;
  document.querySelectorAll("[role=tab]").forEach((button) => {
    const selected = button.dataset.tab === key;
    button.setAttribute("aria-selected", selected);
    button.tabIndex = selected ? 0 : -1;
    $(button.dataset.tab).hidden = !selected;
  });
  loadTab().catch((error) => notice($("page-message"), error.message, true));
}
document.querySelectorAll("[role=tab]").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...document.querySelectorAll("[role=tab]")];
    const index =
      (buttons.indexOf(button) +
        (event.key === "ArrowRight" ? 1 : -1) +
        buttons.length) %
      buttons.length;
    buttons[index].focus();
    showTab(buttons[index].dataset.tab);
  });
});
document.querySelectorAll("[data-export]").forEach((button) =>
  button.addEventListener("click", () => {
    try {
      exportChart(button.dataset.export, activeTab === "parameters" ? segmentationRun() : currentRun());
    } catch (error) {
      notice($("page-message"), error.message, true);
    }
  }),
);

function clearAnalysis() {
  clearAlgorithm($("algorithm-trace"));
  $("model-details").replaceChildren();
  $("result-summary").replaceChildren();
  $("process-export").hidden = true;
  $("model-info").textContent = "";
  for (const [id, message] of [
    ["original-text", "运行分析后显示原始文本与实体高亮。"],
    ["bmes-result", "运行分析后显示字符与 BMES 信息。"],
    ["tokens-result", "运行分析后显示分词结果。"],
    ["ner-result", "状态：等待分析。"],
  ]) {
    $(id).replaceChildren(node("span", message));
    $(id).className = "result-empty";
  }
}

async function loadProcess() {
  const text = $("input-text").value, token = generation;
  if (!text) return;
  if (Array.from(text).length > 30) {
    clearAlgorithm($("algorithm-trace"));
    $("algorithm-trace").append(node("p", "算法过程可视化仅对 ≤30 字符输入开放。当前文本仍已正常完成分词。", "process-note"));
    await loadCurrentModelInfo(null, token);
    return;
  }
  try {
    const trace = await api(`/api/v1/experiments/1/trace?text=${encodeURIComponent(text)}&model=${encodeURIComponent($("segmentation-model").value)}&run_id=${encodeURIComponent(currentRun() || "latest")}`);
    if (token !== generation) return;
    renderTrace($("algorithm-trace"), trace);
    $("algorithm-title").textContent = `${$("segmentation-model").selectedOptions[0].textContent} · 算法过程`;
    $("process-export").hidden = !["bigram", "bilstm-crf"].includes($("segmentation-model").value);
    await loadCurrentModelInfo(trace, token);
  } catch (error) { notice($("page-message"), error.message, true); }
}
async function loadCurrentModelInfo(trace, token) {
  const model = $("segmentation-model").value;
  const value = await api(`/api/v1/models/hmm/parameters?${query()}&model=${encodeURIComponent(model)}&topk=20`);
  if (token !== generation) return;
  renderModelInfo($("model-details"), value, model, trace);
}
$("trace-button").addEventListener("click", loadProcess);
$("process-export").addEventListener("click", () => exportProcess($("interactive"), segmentationRun()));
function entityHighlight(text, entities, target) {
  const chars = Array.from(text); // Python 索引按 Unicode 字符；不能用 JS UTF-16 slice。
  target.replaceChildren();
  target.className = "original";
  let position = 0;
  for (const entity of [...entities].sort((a, b) => a.start - b.start)) {
    if (entity.start < position) continue;
    target.append(
      document.createTextNode(chars.slice(position, entity.start).join("")),
    );
    const span = node(
      "mark",
      chars.slice(entity.start, entity.end).join(""),
      `entity ${entity.type.toLowerCase()}`,
    );
    span.title = `${entity.type} [${entity.start}, ${entity.end})`;
    target.append(span);
    position = entity.end;
  }
  target.append(document.createTextNode(chars.slice(position).join("")));
}
$("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const tasks = ["segmentation", "ner"].filter(
    (task) => $(`task-${task}`).checked,
  );
  if (!tasks.length)
    return notice($("page-message"), "请选择至少一个分析任务。", true);
  const token = generation,
    button = $("analyze-button");
  button.disabled = true;
  button.textContent = "分析中…";
  notice($("page-message"));
  clearAnalysis();
  try {
    const value = await api("/api/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: $("input-text").value,
        tasks,
        run_id: currentRun() || "latest",
        segmentation_model: $("segmentation-model").value,
      }),
    });
    if (token !== generation) return;
    const entities = value.ner?.status === "ok" ? value.ner.entities : [];
    entityHighlight(value.text, entities, $("original-text"));
    const models = [value.segmentation?.model, value.ner?.model].filter(
      Boolean,
    );
    $("model-info").textContent =
      `${models.join(" · ")} · ${value.elapsed_ms} ms`;
    if (value.segmentation) {
      $("bmes-title").textContent = ["longest", "bigram"].includes($("segmentation-model").value) ? "字符与 Derived BMES（由 tokens 派生）" : "字符与 BMES（模型解码路径）";
      const items = [["模型", value.segmentation.model], ["字符数", Array.from(value.text).length], ["词数", value.segmentation.tokens.length], ["Roundtrip", value.segmentation.tokens.join("") === value.text ? "✓ 原文一致" : "不一致"], ["耗时", value.elapsed_ms == null ? "未重测" : `${value.elapsed_ms} ms`]];
      $("result-summary").replaceChildren(...items.map(([label,text])=>{const item=node("span",label);item.append(node("strong",String(text)));return item;}));
      $("bmes-result").className = "bmes";
      $("bmes-result").replaceChildren();
      Array.from(value.text).forEach((character, index) => {
        const cell = node("span", "", "bmes-cell");
        cell.append(
          node(
            "span",
            character === "\n" ? "↵" : character === "\t" ? "⇥" : character,
          ),
          node("small", value.segmentation.tags[index]),
        );
        $("bmes-result").append(cell);
      });
      $("tokens-result").className = "tokens";
      $("tokens-result").replaceChildren(
        ...value.segmentation.tokens.map((token) =>
          node("span", token, "token"),
        ),
      );
    } else {
      $("bmes-result").textContent = $("tokens-result").textContent =
        "未请求分词任务。";
    }
    $("ner-result").replaceChildren();
    if (value.ner?.status === "ok") {
      $("ner-result").className = entities.length
        ? "ner-items"
        : "result-empty";
      entities.forEach((entity) => {
        const row = node("div", "", "ner-item");
        row.append(
          node(
            "span",
            `${entity.text} · ${entity.type}`,
            `entity ${entity.type.toLowerCase()}`,
          ),
          node("small", `[${entity.start}, ${entity.end})`),
        );
        $("ner-result").append(row);
      });
      if (!entities.length)
        $("ner-result").textContent = "模型未识别到 PER / LOC / ORG。";
    } else {
      $("ner-result").className = "result-empty";
      $("ner-result").textContent = value.ner?.error || "未请求 NER 任务。";
    }
    if (value.segmentation) await loadProcess();
  } catch (error) {
    if (token === generation) notice($("page-message"), error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "开始分析";
  }
});

async function loadTab() {
  notice($("page-message"));
  if (!currentRun()) {
    resetCharts();
    if (activeTab !== "interactive")
      notice(
        $("page-message"),
        "尚无实验 Run。请先训练并在留出集评测，或执行显式 demo。",
      );
    return;
  }
  const run = currentRun(),
    token = generation;
  if (activeTab === "parameters") {
    const topk = Number($("topk").value),
      state = $("emission-state").value;
    await loadParameters(
      run,
      topk,
      state,
      () =>
        token === generation &&
        activeTab === "parameters" &&
        topk === Number($("topk").value) &&
        state === $("emission-state").value,
        $("segmentation-model").value,
    );
  }
  if (activeTab === "metrics")
    await loadMetrics(
      run,
      () => token === generation && activeTab === "metrics",
    );
  if (activeTab === "metrics")
    await loadComparison(() => token === generation && activeTab === "metrics");
  if (activeTab === "errors") await loadErrors(run, token);
  resizeCharts();
}
function downloadLinks() {
  for (const type of ["metrics", "predictions", "errors", "config"])
    $(`${type}-download`).href =
      `/api/v1/experiments/1/export/${type}?${query()}`;
}
async function loadErrors(run, token) {
  const value = await api(
    `/api/v1/experiments/1/errors?run_id=${encodeURIComponent(run)}&page=${errorPage}&category=${encodeURIComponent($("error-category").value)}`,
  );
  if (token !== generation || run !== currentRun()) return;
  $("error-list").replaceChildren();
  if (!value.items.length)
    $("error-list").append(
      node("div", "当前 Run 与类别下没有分词错误案例。", "result-empty"),
    );
  for (const item of value.items) {
    if (item.evaluation_task === "ner") {
      const article = node("article", "", "error-case");
      article.append(node("h3", `${item.sample_id} · NER exact-span`), node("p", item.text));
      for (const [label, entities] of [["Gold NER", item.ner_gold], ["预测 NER", item.ner_prediction]]) {
        article.append(node("p", `${label}: ${entities.map((entity) => `${entity.text} / ${entity.type} [${entity.start},${entity.end})`).join(" · ") || "无实体"}`));
      }
      article.append(node("p", `分词预测: ${item.segmentation.tokens.join(" / ")}（无 CWS Gold）`));
      $("error-list").append(article);
      continue;
    }
    const article = node("article", "", "error-case"),
      header = node("header"),
      controls = node("div", "", "header-controls manual-controls");
    header.append(
      node(
        "h3",
        `${item.sample_id} · ${item.manual_category || item.category}`,
      ),
    );
    const select = node("select");
    select.setAttribute("aria-label", `人工类别 ${item.sample_id}`);
    const automatic = node("option", "使用自动类别");
    automatic.value = "";
    select.append(automatic);
    categories.forEach((category) => {
      const option = node("option", category);
      option.value = category;
      select.append(option);
    });
    select.value = item.manual_category || "";
    select.disabled = value.run.read_only || value.run.run_type === "frozen-final-benchmark";
    const label = node("label", "人工类别");
    controls.append(label, select);
    header.append(controls);
    select.addEventListener("change", async () => {
      select.disabled = true;
      try {
        await api(
          `/api/v1/experiments/1/errors/${encodeURIComponent(item.sample_id)}?run_id=${encodeURIComponent(run)}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category: select.value || null }),
          },
        );
        await loadErrors(run, token);
      } catch (error) {
        notice($("page-message"), error.message, true);
      } finally {
        select.disabled = false;
      }
    });
    const dl = node("dl");
    const original = node("dd");
    const chars = Array.from(item.text);
    let position = 0;
    for (const [start, end] of item.error_spans) {
      original.append(
        document.createTextNode(chars.slice(position, start).join("")),
        node("mark", chars.slice(start, end).join(""), "error-highlight"),
      );
      position = end;
    }
    original.append(document.createTextNode(chars.slice(position).join("")));
    dl.append(node("dt", "原文"), original);
    for (const [label, text] of [
      ["Gold 分词", item.gold_tokens.join(" / ")],
      ["预测分词", item.predicted_tokens.join(" / ")],
      ["Gold BMES", item.gold_tags.join(" ")],
      ["预测 BMES", item.predicted_tags.join(" ")],
      [
        "错误跨度",
        item.error_spans.map((span) => `[${span.join(", ")})`).join(" · "),
      ],
      [
        "自动判定",
        `${item.category} · heuristic${item.category_candidates.length ? ` · 候选：${item.category_candidates.join(" / ")}` : " · 可人工标记 Ambiguity 等类别"}`,
      ],
    ])
      dl.append(node("dt", label), node("dd", text));
    article.append(header, dl);
    $("error-list").append(article);
  }
  $("errors-page").textContent =
    `第 ${errorPage} 页 · 共 ${value.total} 个案例`;
  $("errors-prev").disabled = errorPage <= 1;
  $("errors-next").disabled = errorPage * value.page_size >= value.total;
}
$("errors-prev").addEventListener("click", () => {
  errorPage--;
  loadTab().catch((error) => notice($("page-message"), error.message, true));
});
$("errors-next").addEventListener("click", () => {
  errorPage++;
  loadTab().catch((error) => notice($("page-message"), error.message, true));
});
$("error-category").addEventListener("change", () => {
  errorPage = 1;
  loadTab().catch((error) => notice($("page-message"), error.message, true));
});
for (const id of ["topk", "emission-state"])
  $(id).addEventListener("change", () =>
    loadTab().catch((error) => notice($("page-message"), error.message, true)),
  );
$("run-select").addEventListener("change", () => {
  generation++;
  errorPage = 1;
  clearAnalysis();
  resetCharts();
  const option = $("run-select").selectedOptions[0];
  updateActiveRun();
  $("run-warning").hidden = option?.dataset.demo !== "true";
  $("run-warning").className = "notice warning";
  $("run-warning").textContent =
    "演示数据：仅验证系统流程，不代表正式课程实验性能。";
  downloadLinks();
  loadTab().catch((error) => notice($("page-message"), error.message, true));
});
$("segmentation-model").addEventListener("change", () => {
  generation++;
  clearAnalysis();
  updateActiveRun();
  loadTab().catch((error) => notice($("page-message"), error.message, true));
});
window.addEventListener("resize", () => { resizeCharts(); resizeAlgorithm(); });
async function initialize() {
  const [value, extensionValue] = await Promise.all([
    api("/api/v1/experiments/1/runs"),
    api("/api/v1/experiments/1/extensions"),
  ]);
  for (const item of extensionValue.items || []) extensionRuns.set(item.model, item);
  const available = new Set((extensionValue.items || []).filter((item) => item.available !== false).map((item) => item.model));
  document.querySelectorAll("#segmentation-model option").forEach((option) => {
    if (option.value === "hmm1-smooth") return;
    if (!available.has(option.value)) {
      option.disabled = true;
      option.textContent += "（unavailable）";
    }
  });
  if (!value.items.length) {
    resetCharts();
    return;
  }
  $("run-select").replaceChildren(
    ...value.items.map((run) => {
      const option = node(
        "option",
        `${run.is_demo ? "演示 · " : ""}${run.dataset} · ${run.run_id}`,
      );
      option.value = run.run_id;
      option.dataset.demo = run.is_demo;
      option.dataset.model = run.model;
      return option;
    }),
  );
  $("run-select").value = value.latest;
  $("run-select").dispatchEvent(new Event("change"));
}
initialize().catch((error) => notice($("page-message"), error.message, true));
